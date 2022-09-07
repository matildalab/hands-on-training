# Getting Started

This tutorial introduces basic user guide for IPU beginners. It contains how to setup environments initially to properly use IPUs and basic porting guide for Pytorch and Tensorflow.
Note that this tutorial is made for hands-on sessions for Graphcore customers in Korea. To see official Graphcore user guide, please visit our [official documentations](https://docs.graphcore.ai/en/latest/#).

The first thing you might want to check is to see how many IPUs you currently have. This can be done by a command from [Graphcore command line tools](https://docs.graphcore.ai/projects/command-line-tools/en/latest/index.html).  
Command:
```bash
$ gc-monitor
```
Output:
```
bash: gc-monitor: command not found
```
At the first time, you will see the above message. This appears when Poplar SDK is not enabled. It is essential to activate the SDK whenever you try to use IPUs. You can enable Poplar SDK by running `enable.sh` included in the SDK.

Command:
```bash
$ source [SDK-path]/poplar*/enable.sh
$ source [SDK-path]/popart*/enable.sh
```
It is an essential step everytime you login, so it might be more convenient to include them in `.bashrc`, instead of running the commands everytime. If you have no SDK downloaded on the server, you can download the latest SDK on [our support site](https://www.graphcore.ai/support). Please request to Field Engineers for an account to download it if you don't have one.

Once the SDK is properly activated, you can see the IPUs you have in the current partition.

Output:
```
+---------------+---------------------------------------------------------------------------------+
|  gc-monitor   |            Partition: example-partition [active] has 16 reconfigurable IPUs     |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|    IPU-M    |       Serial       |IPU-M SW|Server version|  ICU FW  | Type  | ID | IPU# |Routing|
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|  10.1.5.21  | 0046.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 0  |  3   |  DNC  |
|  10.1.5.21  | 0046.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 1  |  2   |  DNC  |
|  10.1.5.21  | 0046.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 2  |  1   |  DNC  |
|  10.1.5.21  | 0046.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 3  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|  10.1.5.22  | 0012.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 4  |  3   |  DNC  |
|  10.1.5.22  | 0012.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 5  |  2   |  DNC  |
|  10.1.5.22  | 0012.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 6  |  1   |  DNC  |
|  10.1.5.22  | 0012.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 7  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|  10.1.5.23  | 0040.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 8  |  3   |  DNC  |
|  10.1.5.23  | 0040.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 9  |  2   |  DNC  |
|  10.1.5.23  | 0040.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 10 |  1   |  DNC  |
|  10.1.5.23  | 0040.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 11 |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
|  10.1.5.24  | 0050.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 12 |  3   |  DNC  |
|  10.1.5.24  | 0050.0002.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 13 |  2   |  DNC  |
|  10.1.5.24  | 0050.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 14 |  1   |  DNC  |
|  10.1.5.24  | 0050.0001.8204521  |        |    1.9.0     |  2.4.4   | M2000 | 15 |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+-------+----+------+-------+
+--------------------------------------------------------------------------------------------------+
|                           No attached processes in partition example-partition                   |
+--------------------------------------------------------------------------------------------------+
```
You can also check the version of Poplar SDK currently activated.
```bash
$ popc --version
```
```
POPLAR version 2.6.0 (e0ab3b4f12)
clang version 15.0.0 (765f0d82adc5d9261ca9d1f86d11b594a33bd784)
```
Now, you are ready to run programs on IPU.

[Continue to the next chapter](../2.%20Running%20Pytorch%20on%20IPU)