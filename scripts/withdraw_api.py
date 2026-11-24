#!/usr/bin/env python3
"""Withdraw one API from its AI_LOOP batch without publishing or changing the database."""
import argparse
from contextlib import ExitStack, closing
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys

from npu_handoff import file_hash, record_lock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'AI_LOOP/scripts'))
from branch_mapping import branch_group, grouped_mapping
from preparation_state import locked, write_json
from prepare_development_branch import current_branch, read_object
from pr_tracking import file_lock
from publish_pr import issue_lock
from reconcile_prs import resolve_rows
from refresh_api_support_db import NIGHTLY_TABLE


def contained(root, name):
    """Reject symlinks, traversal and absolute artifact names, including on retries."""
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError(f'Invalid relative path: {name}')
    path = root / rel
    if path.resolve() != path:
        raise ValueError(f'Unexpected symlink: {path}')
    return path


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def api_names(row):
    return {row[k] for k in ('api', 'database_api', 'api_full_name') if row.get(k)}


def file_references(obj):
    """Find paths named by retained rows, including manifest keys and helper files."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for item in obj:
            yield from file_references(item)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from file_references(value)


def source_files(repo, baseline, rows, kept_refs, names):
    remove, review, shared = set(), set(), set()
    expected = set()
    for name in names:
        full = 'torch.' + name if name == 'Tensor' or name.startswith('Tensor.') else name
        if all(s.isidentifier() for s in full.split('.')):
            expected.add('test/api_supplement/' + full.replace('.', '/') + '/test_' + full.replace('.', '_') + '.py')
    for row in rows:
        npu = row.get('device_validation', {}).get('npu', {})
        for rel in row.get('test_files', []):
            path = contained(repo, rel)
            if not path.is_file():
                continue
            if rel in kept_refs:
                shared.add(rel)
                continue
            if git(repo, 'ls-tree', baseline, '--', rel):
                if git(repo, 'diff', '--name-only', baseline, '--', rel):
                    review.add(rel)
                continue
            if rel not in expected:
                review.add(rel)
                continue
            digest = npu.get('handoff_files', {}).get(rel)
            if not digest or file_hash(path) != digest:
                review.add(rel)
                continue
            # A surviving caller can import this test module as a helper.
            refs = subprocess.run(['git', '-C', str(repo), 'grep', '--untracked', '-l', '-F',
                                   path.stem, '--', '*.py'], capture_output=True, text=True)
            if refs.returncode not in (0, 1):
                raise ValueError(refs.stderr)
            if set(refs.stdout.splitlines()) - {rel}:
                shared.add(rel)
            else:
                remove.add(path)
    return remove, review, shared


def normalized_refs(rows, root):
    refs = set(file_references(rows))
    for ref in list(refs):
        if '/' not in ref or '\n' in ref or '\x00' in ref:
            continue
        path = Path(ref)
        if path.is_absolute() and path.is_relative_to(root):
            refs.add(path.relative_to(root).as_posix())
        elif not path.is_absolute():
            refs.add(path.as_posix())
    return refs


def run_material(run, rows, retained, names):
    loop = run / 'api-loop'
    refs = normalized_refs(retained, run)
    delete, updates, reexport, shared = set(), {}, [], set()
    attempts = set()
    for row in rows:
        design = row.get('adaptation', {})
        if design.get('design'):
            f = contained(run, design['design'])
            if (f.is_relative_to(loop / 'design') and f.parent.name in names and f.suffix == '.md'
                    and f.relative_to(run).as_posix() not in refs and f.is_file() and file_hash(f) == design.get('sha256')):
                delete.add(f)
        npu = row.get('device_validation', {}).get('npu', {})
        # Only remove files belonging to a verified, single-API NPU attempt.
        snap_name = npu.get('snapshot')
        if not snap_name:
            continue
        snap = contained(run, snap_name)
        if not snap.is_relative_to(loop / 'npu-attempts') or not snap.is_file():
            continue
        attempts.add(snap)
        if read_object(snap).get('api') != row['api']:
            raise ValueError(f'NPU attempt has a different owner: {snap}')
        if any(ref.startswith(str(snap.parent.relative_to(run)) + '/') for ref in refs):
            shared.add(str(snap.parent.relative_to(run)))
            continue
        for rel, digest in npu.get('artifacts', {}).items():
            f = contained(run, rel)
            if f.parent != snap.parent or rel in refs or f.suffix not in {'.json', '.md', '.log', '.txt'}:
                continue
            if f.is_file() and file_hash(f) == digest:
                delete.add(f)
    # Retries and pre-fix runs can have snapshots that result.json no longer
    # points to. Inspect the recorded API and file scope, not directory names.
    for snap in (loop / 'npu-attempts').rglob('snapshot.json'):
        contained(run, snap.relative_to(run))
        old = read_object(snap)
        if old.get('api') not in names or not old.get('test_files') or 'started_ns' not in old:
            continue
        directory = snap.parent
        prefix = directory.relative_to(run).as_posix() + '/'
        if any(ref.startswith(prefix) for ref in refs) or set(old['test_files']).intersection(refs):
            shared.add(prefix.rstrip('/'))
            delete.difference_update(f for f in list(delete) if f.parent == directory)
            continue
        standard = ['run-exit.txt']
        if snap not in attempts:
            standard += ['snapshot.json', 'UT_REPORT.md', 'report.md', 'test.log', 'run.log']
        for name in standard:
            f = contained(directory, name)
            if f.is_file():
                delete.add(f)
    for folder, filename in [('adaptation', 'before.json'), ('repairs', 'previous-result.json')]:
        for f in (loop / folder).glob('*/' + filename):
            contained(run, f.relative_to(run))
            payload = read_object(f)
            owner = payload.get('result', payload)
            if api_names(owner) & names and f.relative_to(run).as_posix() not in refs:
                delete.add(f)
    # Each GPU manifest binds the whole export. Mixed reports remain byte-for-byte
    # intact for the other APIs. A single-API export can retire its known payloads.
    for manifest in (loop / 'reports').rglob('manifest.json'):
        contained(run, manifest.relative_to(run))
        obj = read_object(manifest)
        scope = {v['api'] for v in obj.get('results', []) if isinstance(v, dict) and 'api' in v}
        if not scope.intersection(names):
            continue
        directory = manifest.parent
        prefix = directory.relative_to(run).as_posix() + '/'
        if scope - names or any(ref.startswith(prefix) for ref in refs):
            shared.add(prefix.rstrip('/'))
            continue
        report = directory / 'evidence/report.json'
        if report.exists():
            contained(run, report.relative_to(run))
            actual = read_object(report)
            if actual.get('manifest_sha256') != file_hash(manifest) or {
                    entry['api'] for entry in actual.get('results', [])} != scope:
                shared.add(prefix.rstrip('/'))
                continue
        for rel in ('manifest.json', 'bundle.zip', 'evidence/report.json', 'submission.json', 'run.json'):
            f = contained(directory, rel)
            if f.is_file():
                delete.add(f)
        if report.is_file():
            for f in report.parent.rglob('*'):
                if f.name in {'junit.xml', 'step1-audit.json', 'selected-files.txt', 'test-output.tar.gz',
                              'environment.json', 'probe.log', 'install.log', 'gpu-preflight.log'} or re.fullmatch(
                                  r'api-\d+-file-\d+\.(json|log)', f.name):
                    contained(directory, f.relative_to(directory))
                    if f.is_file():
                        delete.add(f)
    for f in (loop / 'handoffs').glob('*.json'):
        contained(run, f.relative_to(run))
        if api_names(read_object(f)) & names and str(f.relative_to(run)) not in refs:
            delete.add(f)
    feedback = contained(run, 'api-loop/classification-feedback.json')
    if feedback.exists():
        content = json.loads(feedback.read_bytes())
        if isinstance(content, list):
            rest = [item for item in content if not api_names(item) & names]
            if rest != content:
                updates[feedback] = rest
        elif api_names(content) & names:
            delete.add(feedback)
    for f in loop.rglob('step1-request.json'):
        contained(run, f.relative_to(run))
        ticket = read_object(f)
        if not names.intersection(ticket.get('apis', [])):
            continue
        if ticket.get('status') == 'running':
            raise ValueError(f'Finish or cancel the active test before withdrawal: {f}')
        if ticket.get('status') == 'needs_workstation':
            delete.add(f)
            reexport.extend(n for n in ticket['apis'] if n not in names)
        elif ticket.get('status') == 'result_received' and set(ticket['apis']) <= names and f.with_name('manifest.json') in delete:
            delete.add(f)
    for f in run.glob('*.json'):
        # Export configurations point at this batch result. Receipts and historical
        # reports have different schemas and are left as evidence of the actual run.
        if f.name in {'adapt-information.json', 'basic_information.json', 'pr.json', 'issue.json',
                      'issue-publish-state.json', 'issue-search.json', 'publish-state.json',
                      'review-handling.json', 'pr-review-comments.json'} or f.is_symlink():
            continue
        try:
            obj = json.loads(f.read_bytes())
        except (ValueError, UnicodeError):
            continue  # Unrelated scratch files are not configuration inputs.
        if not isinstance(obj, dict) or not obj.get('source_root') or not isinstance(obj.get('results'), str):
            continue
        if (run / obj['results']).resolve() != loop / 'result.json':
            continue
        for key in ('apis', 'device_independent_apis'):
            if isinstance(obj.get(key), list) and names.intersection(obj[key]):
                obj[key] = [v for v in obj[key] if v not in names]
                updates[f] = obj
        if 'apis' in obj and not obj['apis']:
            updates.pop(f, None)
            delete.add(f)
    return delete, updates, sorted(set(reexport)), sorted(shared)


def withdraw(run, api, source=None, dry=False):
    run = run.resolve()
    if run.parent.name != 'run_data' or run.parent.parent.name != 'data':
        raise ValueError('Expected AI_LOOP/data/run_data/<run-id>')
    data = run.parent.parent
    root = data.parent
    adapt_path = contained(run, 'adapt-information.json')
    result_path = contained(run, 'api-loop/result.json')
    mapping_path = contained(data, 'branch-mapping.json')
    with ExitStack() as stack:
        stack.enter_context(locked(data / 'source_code/.torchnpu-source.lock'))
        issue = run / 'issue.json'
        if issue.is_file():
            stack.enter_context(issue_lock(run, read_object(issue)['url'].rstrip('/').lower()))
        stack.enter_context(file_lock(data / '.pr-tracking.lock', wait=True))
        stack.enter_context(record_lock(result_path))
        basic = read_object(run / 'basic_information.json')
        adapt = read_object(adapt_path)
        control = adapt['git-control']
        branch = adapt['development-branch']
        group = branch_group(basic, root)
        if control.get('branch-group', group) != group or control.get('branch') != branch or control.get('status') != 'BRANCH_READY':
            raise ValueError('Withdrawal requires the prepared development branch for this batch')
        table = NIGHTLY_TABLE if group == 'nightly' else NIGHTLY_TABLE + '_' + group
        db = stack.enter_context(closing(sqlite3.connect((data / 'api-support.sqlite3').as_uri() + '?mode=ro', uri=True)))
        db.row_factory = sqlite3.Row
        canonical = resolve_rows(db, [api], table)[api]['API名称']
        excluded = adapt.get('withdrawn-apis', [])
        pending = adapt.get('withdrawing-api')
        if pending and pending != canonical:
            raise ValueError(f'Resume the incomplete withdrawal of {pending} first')
        if canonical in excluded and not pending:
            return {'status': 'ALREADY_WITHDRAWN', 'api': canonical}
        receipt_path = run / 'pr.json'
        receipt = read_object(receipt_path) if receipt_path.exists() else {}
        if receipt:
            if receipt.get('cleanup-completed') or receipt.get('head') != branch:
                raise ValueError('PR already reconciled or branch does not match')
            index = data / 'pr-tracking.json'
            if index.exists():
                states = [v['pr_status'] for v in json.loads(index.read_bytes())
                          if v['run_data'] == run.relative_to(root).as_posix()]
                if any(v in ('merged', 'merged_pending', 'closed') for v in states):
                    raise ValueError('PR is already merged or closed')
        task_list = adapt['task-branches']
        tasks = [t for t in task_list if t.startswith('adapt-')]
        resolved = resolve_rows(db, (t[6:] for t in tasks), table)
        owned = {t for t in tasks if resolved[t[6:]]['API名称'] == canonical}
        if not owned:
            raise ValueError('API is not owned by this batch: ' + api)
        original = (root / basic['torch_npu_latest_source']).resolve()
        if original != (data / 'source_code/torch_npu').resolve():
            raise ValueError('Unexpected Step 3 source repository')
        repo = Path(source or original).resolve()
        if (Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() != repo or current_branch(repo) != branch
                or git(original, 'rev-parse', '--path-format=absolute', '--git-common-dir') !=
                   git(repo, 'rev-parse', '--path-format=absolute', '--git-common-dir')):
            raise ValueError('Select the worktree for this batch and development branch')
        baseline = control['base-commit']
        if not re.fullmatch(r'[a-fA-F0-9]{40,64}', baseline):
            raise ValueError('Invalid Step 3 base commit')
        git(repo, 'cat-file', '-e', baseline + '^{commit}')
        names = {api, canonical} | {t[6:] for t in owned}
        status_rows = resolve_rows(db, adapt.get('api-status', {}), table)
        names.update(n for n, v in status_rows.items() if v['API名称'] == canonical)
        result = read_object(result_path)
        rows, kept = [], []
        for row in result['results']:
            if api_names(row) & names:
                stored = row.get('database_api', row['api'])
                if resolve_rows(db, [stored], table)[stored]['API名称'] != canonical:
                    raise ValueError('Conflicting database identity in API results: ' + stored)
                rows.append(row)
            else:
                kept.append(row)
        for row in rows:
            names.update(api_names(row))
        if any(api_names(row) & names for row in kept):
            raise ValueError('Ambiguous API identity among retained results')
        protected = normalized_refs(kept, repo)
        source_delete, review, shared_tests = source_files(repo, baseline, rows, protected, names)
        missing_owners = {resolved[t[6:]]['API名称'] for t in tasks if t not in owned} - {
            row.get('database_api', row['api']) for row in kept}
        if missing_owners:
            review.update(f.relative_to(repo).as_posix() for f in source_delete)
            source_delete.clear()
        inherited = control.get('source-branches', [])
        for f in list(source_delete):
            rel = f.relative_to(repo).as_posix()
            if any(git(repo, 'ls-tree', 'refs/heads/' + ref, '--', rel) for ref in inherited):
                source_delete.remove(f)
                shared_tests.add(rel)
        delete, updates, reexport, shared = run_material(run, rows, kept, names)
        if missing_owners:
            uncertain = set()
            for f in delete:
                if (f.is_relative_to(run / 'api-loop/reports') or f.is_relative_to(run / 'api-loop/npu-attempts')) and f.name != 'step1-request.json':
                    uncertain.add(f)
            shared = sorted(set(shared) | {str(f.relative_to(run)) for f in uncertain})
            delete -= uncertain
        groups = grouped_mapping(read_object(mapping_path))
        mapping = groups.get(group, {})
        removed_maps = sorted(t for t in owned if mapping.get(t) == branch)
        remaining = [t for t in task_list if t not in owned]
        drafts = [n for n in ('issue-title.txt','issue-body.md','pr-title.txt','pr-body.md') if contained(run,n).is_file()]
        pattern = re.compile(r'(?<![\w.])(?:' + '|'.join(re.escape(n) for n in names) + r')(?![\w.])')
        stale_drafts = [n for n in drafts if pattern.search((run / n).read_text(encoding='utf-8-sig'))]
        if not remaining:
            delete.update(run / name for name in drafts)
        implementation = git(repo, 'diff', '--name-only', baseline, '--', '.', ':(exclude)test/api_supplement/**').splitlines()
        implementation += git(repo, 'ls-files', '--others', '--exclude-standard', '--', '.',
                              ':(exclude)test/api_supplement/**').splitlines()
        summary = dict(api=canonical, remaining=remaining,
                       remove_tests=sorted(str(f.relative_to(repo)) for f in source_delete),
                       remove_material=sorted(str(f.relative_to(run)) for f in delete),
                       update_material=sorted(str(f.relative_to(run)) for f in updates),
                       preserve_shared_evidence=shared, review_source=sorted(review), shared_tests=sorted(shared_tests),
                       review_implementation=sorted(set(implementation)),
                       remove_mappings=removed_maps, reexport_apis=reexport,
                       refresh_drafts=drafts, drafts_naming_api=stale_drafts, refresh_original_pr=bool(receipt))
        if dry:
            return dict(status='WITHDRAWAL_PREVIEW', **summary)
        if remaining and stale_drafts:
            raise ValueError('Regenerate shared draft text and statistics for retained APIs: ' + ', '.join(stale_drafts))
        if review:
            raise ValueError('Resolve changed or unowned test files before withdrawal: ' + ', '.join(sorted(review)))
        stamps = {f: f.stat() for f in delete | source_delete}
        # Durable intent precedes destructive changes. Keep the original task list
        # until all other writers finish; the same command can resume an interruption.
        adapt['withdrawing-api'] = canonical
        write_json(adapt_path, adapt)
        for f, value in updates.items():
            write_json(f, value)
        for f in sorted(delete | source_delete, key=lambda path: (path.name in {'manifest.json','snapshot.json'}, str(path))):
            current = f.lstat()
            before = stamps[f]
            if (current.st_ino, current.st_size, current.st_mtime_ns) != (before.st_ino, before.st_size, before.st_mtime_ns):
                raise ValueError(f'File changed during withdrawal: {f}')
            f.unlink()
        if removed_maps:
            for task in removed_maps:
                del mapping[task]
            write_json(mapping_path, groups)
        result['results'] = kept
        write_json(result_path, result)
        adapt['task-branches'] = summary['remaining']
        for step in adapt.get('adapt-plan', {}).values():
            step['task-branches'] = [t for t in step['task-branches'] if t not in owned]
        for key in ('requested-branches', 'remaining-branches', 'existing-branches'):
            if key in control:
                control[key] = [t for t in control[key] if t not in owned]
        for t in owned:
            # A newly created task's empty pre-claim must not prevent a different
            # batch claiming it later. Retain snapshots of actual inherited branches.
            if control.get('original-mappings', {}).get(t) is None:
                control.get('original-mappings', {}).pop(t, None)
        for name in names:
            adapt.get('api-status', {}).pop(name, None)
        adapt['withdrawn-apis'] = sorted(set(excluded) | {canonical})
        del adapt['withdrawing-api']
        write_json(adapt_path, adapt)
        return dict(status='API_WITHDRAWN', **summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--api', required=True)
    parser.add_argument('--source', type=Path, help='Batch branch in an alternate worktree')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        print(json.dumps(withdraw(args.run_dir, args.api, args.source, args.dry_run), ensure_ascii=False, indent=2))
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error, subprocess.CalledProcessError) as error:
        parser.exit(1, f'WITHDRAWAL_FAILED: {error}\n')


if __name__ == '__main__':
    main()
