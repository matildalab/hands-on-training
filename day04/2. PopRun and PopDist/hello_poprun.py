import popdist

if __name__ == '__main__':
    if popdist.isPopdistEnvSet():
        message = f'''poprun is invoked with "--num-replicas {popdist.getNumTotalReplicas()}", "--num-instances {popdist.getNumInstances()}" and "--ipus-per-replica {popdist.getNumIpusPerReplica()}".
        - popdist.getInstanceIndex(): {popdist.getInstanceIndex()}.
        - popdist.getNumLocalReplicas(): {popdist.getNumLocalReplicas()}.
        - popdist.getReplicaIndexOffset(): {popdist.getReplicaIndexOffset()}.
        - popdist.getDeviceId(): {popdist.getDeviceId()}.
        (To see the IPU chip IDs in this device, run "gc-info -d {popdist.getDeviceId()} --chip-id")'''
        print(message)
    else:
        print(f'poprun is not invoked.')
