# 2. Command Line Tools
In this chapter, you will be introduced with several important command line tools that come with the Poplar SDK.
You can find the full documentation for the command line tools in the document [Graphcore Command Line Tools](https://docs.graphcore.ai/projects/command-line-tools/en/latest/index.html).

## gc-info
`gc-info` is a multi-function tool that can display various kinds of information about the available IPUs.

Here are some useful commands and options.

* `-l` or `--list-devices`: The `--list-devices` command displays a list of all devices in the active partition. This includes all individual IPUs and all logical devices composed of multiple IPUs.

	<details><summary><strong>Example</strong></summary><p>

	```
	$ gc-info --list-devices # same as gc-info -l
	Graphcore device listing:
	Partition: p1 [active]
	-+- Id: [0], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [3]
	-+- Id: [1], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [2]
	-+- Id: [2], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [1]
	-+- Id: [3], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [0]
	-+- Id: [4], target: [Multi IPU]
	 |--- Id: [0], DNC Id: [0], IPU-M host: [10.1.5.10], IPU#: [3]
	 |--- Id: [1], DNC Id: [1], IPU-M host: [10.1.5.10], IPU#: [2]
	-+- Id: [5], target: [Multi IPU]
	 |--- Id: [2], DNC Id: [0], IPU-M host: [10.1.5.10], IPU#: [1]
	 |--- Id: [3], DNC Id: [1], IPU-M host: [10.1.5.10], IPU#: [0]
	-+- Id: [6], target: [Multi IPU]
	 |--- Id: [0], DNC Id: [0], IPU-M host: [10.1.5.10], IPU#: [3]
	 |--- Id: [1], DNC Id: [1], IPU-M host: [10.1.5.10], IPU#: [2]
	 |--- Id: [2], DNC Id: [2], IPU-M host: [10.1.5.10], IPU#: [1]
	 |--- Id: [3], DNC Id: [3], IPU-M host: [10.1.5.10], IPU#: [0]
	```
	
	</p></details>
	
	You can see what devices are available in other partitions by using the --all-partitions option flag.
	
	<details><summary><strong>Example</strong></summary><p>
	
	```
	gc-info -l --all-partitions
	Graphcore device listing:
	Partition: p1 [active]
	-+- Id: [0], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [3]
	-+- Id: [1], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [2]
	-+- Id: [2], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [1]
	-+- Id: [3], target: [Fabric], IPU-M host: [10.1.5.10], IPU#: [0]
	Partition: p2
	-+- Id: [4], target: [Fabric], IPU-M host: [10.1.5.12], IPU#: [3]
	-+- Id: [5], target: [Fabric], IPU-M host: [10.1.5.12], IPU#: [2]
	-+- Id: [6], target: [Fabric], IPU-M host: [10.1.5.12], IPU#: [1]
	-+- Id: [7], target: [Fabric], IPU-M host: [10.1.5.12], IPU#: [0]
	```
	
	</p></details>

* `-d` or `--device-id`: You can specify one of the IPU device IDs discovered by `gc-info -l` via `--device-id` option to figure out more about a specific IPU device.
	* `--device-info`: The `--device-info` command displays the device attributes of the specified device.
	
	<details><summary><strong>Example</strong></summary><p>
		
	```
	$ gc-info --device-info -d 0
	Device Info:
	id: 0
	target: Fabric
	average board temp: N/A
	average die temp: N/A
	board ipu index: 3
	board serial number: 0114.0002.8204521
	board type: M2000
	clock: 1330MHz
	config domain: 94414229001808
	driver version: 1.0.56
	firmware major version: 2
	firmware minor version: 4
	firmware patch version: 4
	firmware version: 2.4.4
	gateway software version: 2.4.2
	graph streaming: true
	hexoatt active size (bytes): 0
	hexoatt total size (bytes): 34082914304
	hexopt active size (bytes): 0
	hexopt total size (bytes): 268435456
	host link correctable error count: 2932108
	ipu architecture: ipu2
	ipu error state: no errors
	ipu power: N/A
	ipu utilisation: 0.00%
	ipu utilisation (session): 0.00%
	ipuof host: 10.1.5.17
	ipuof partition id: pt-jiwoongc-16-ipus
	ipuof routing id: 0
	ipuof routing type: DNC
	ipuof server version: 1.9.0
	link correctable error count: 0
	link speed: 16 GT/s
	link width: 8
	number of replicas: 1
	partition sync type: c2-compatible
	pci id: 3
	pcie physical slot: 3
	reconfigurable partition: true
	remote buffers supported: 1
	total board power: N/A
	```
	
	</p></details>


	* `--tile-overview`: The `--tile-overview` command will show a representation of the sync state of all the tiles of the IPU or IPUs this device is connected to.
	
	<details><summary><strong>Example</strong></summary><p>
	
	```
	$ gc-info -d 0 --tile-overview
      0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
  0 xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx:: xx::
  1 :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: ::::
  2 :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: ::::
  3 :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: :::: ::::
	```
	
	</p></details>
	
## gc-monitor
You can use this command to monitor IPU activity without affecting users of the IPUs. This can be used to:

* Check and monitor what’s currently running on which IPU in shared systems.
* Make sure code is correctly running on an IPU.
* Monitor performance: the power and temp will increase, and the clock rate will drop when an IPU is heavily loaded.

<details><summary><strong>Example</strong></summary><p>

```
+---------------+--------------------------------------------------------------------------------+
|  gc-monitor   |            Partition: p1 (gcd:0) [active] has 8 reconfigurable IPUs            |
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
|    IPU-M    |       Serial       |IPU-M SW|Server version|  ICU FW  | Type | ID | IPU# |Routing|
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
|  10.1.5.10  | 0024.0002.8203321  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 0  |  3   |  DNC  |
|  10.1.5.10  | 0024.0002.8203321  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 1  |  2   |  DNC  |
|  10.1.5.10  | 0024.0001.8203321  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 2  |  1   |  DNC  |
|  10.1.5.10  | 0024.0001.8203321  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 3  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
|  10.1.5.11  | 0013.0002.8204921  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 4  |  3   |  DNC  |
|  10.1.5.11  | 0013.0002.8204921  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 5  |  2   |  DNC  |
|  10.1.5.11  | 0013.0001.8204921  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 6  |  1   |  DNC  |
|  10.1.5.11  | 0013.0001.8204921  | 2.5.0  |    1.9.0     |  2.4.4   |M2000 | 7  |  0   |  DNC  |
+-------------+--------------------+--------+--------------+----------+------+----+------+-------+
```

</p></details>

By default, IPU applications do not read power and temperature sensors, so this information will not be available in gc-monitor. To enable sensor reading the application must be launched with the GCDA_MONITOR environment variable set.

```
GCDA_MONITOR=1 python main.py
watch -n 1 gc-monitor
```

Here are some useful options.

* --no-card-info: Don’t display card information

<details><summary><strong>Example</strong></summary><p>

```
+------------------------------------------------------------------------+------------------------+-----------------+
|                     Attached processes in partition p1                 |          IPU           |      Board      |
+--------+-----------------------------------------+--------+------------+----+----------+--------+--------+--------+
|  PID   |                 Command                 |  Time  |    User    | ID |  Clock   |  Temp  |  Temp  | Power  |
+--------+-----------------------------------------+--------+------------+----+----------+--------+--------+--------+
| 81816  |           gc-hosttraffictest            |  10s   |  justina   | 14 | 1300MHz  | 33.7 C | 35.0 C |113.7 W |
| 81816  |           gc-hosttraffictest            |  10s   |  justina   | 15 | 1300MHz  | 37.4 C |        |        |
+--------+-----------------------------------------+--------+------------+----+----------+--------+--------+--------+
| 81816  |           gc-hosttraffictest            |  10s   |  justina   | 12 | 1300MHz  | 31.4 C | 32.8 C |107.1 W |
| 81816  |           gc-hosttraffictest            |  10s   |  justina   | 13 | 1300MHz  | 34.7 C |        |        |
+--------+-----------------------------------------+--------+------------+----+----------+--------+--------+--------+
```

</p></details>


* `--all-partitions`: Show information about all partitions (default is active partition only)

## gc-reset
This tool resets the IPU devices. For example:

```
gc-reset -d {device_id}
```

where {device_id} is the id number returned by the `gc-info` tool. This can refer to one IPU or a group of IPUs.
