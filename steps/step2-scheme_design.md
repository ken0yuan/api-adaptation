# Step2：待适配入口

接收Step0直接分流的待适配API，以及Step1补测失败后转来的待适配API。

保留本次`adapt-information.json.api-status`中的`待适配`状态、已有用例和测试材料，向调用方交回当前结果并结束本轮。当前不执行方案设计、代码开发或回归测试。
