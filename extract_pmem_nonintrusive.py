from libraries.lauterbach import *
import time

from multiprocessing import shared_memory
from multiprocessing import Value
import multiprocessing
import numpy as np

from os import listdir
from os.path import isfile, join
import collections

INIT_TASK = 0xc150c340      # For BBB kernel 4.19.82-ti-rt-r31
INT_INIT_TASK = 3243295552
MAX_PROCESS_COUNT = 32768

def GetNextAddress(conn, address):
    return int(conn.HexReadMemory(address + 0x340, 0x40, 0x04), 16) - 0x340

def GetPreviousAddress(conn, address):
    return int(conn.HexReadMemory(address + 0x344, 0x40, 0x04), 16) - 0x340

def GetPID(conn, address):
    return conn.HexReadMemory(address + 0x3d8, 0x40, 0x04)

def GetTaskStruct(conn, address):
    return conn.HexReadMemory(address, 0x40, 0x87f)

def GetMMStructAddress(conn, address):
    mmAddress = address + 0x36c
    return conn.HexReadMemory(mmAddress, 0x40, 0x04)

def GetProcessMemoryAddress(conn, address):
    pgdAddress = address + 0x24
    codeStartAddress = address + 0xac
    codeEndAddress = address + 0xb0
    dataStartAddress = address + 0xb4
    dataEndAddress = address + 0xb8

    brkStartAddress = address + 0xbc # Heap address
    brkEndAddress = address + 0xc0
    argStartAddress = address + 0xc8
    argEndAddress = address + 0xcc
    envStartAddress = address + 0xd0
    envEndAddress = address + 0xd4

    pgdBase = conn.HexReadMemory(pgdAddress, 0x40, 0x04)
    codeStart = conn.HexReadMemory(codeStartAddress, 0x40, 0x04)
    codeEnd = conn.HexReadMemory(codeEndAddress, 0x40, 0x04)
    dataStart = conn.HexReadMemory(dataStartAddress, 0x40, 0x04)
    dataEnd = conn.HexReadMemory(dataEndAddress, 0x40, 0x04)
    brkStart = conn.HexReadMemory(brkStartAddress, 0x40, 0x04)
    brkEnd = conn.HexReadMemory(brkEndAddress, 0x40, 0x04)
    argStart = conn.HexReadMemory(argStartAddress, 0x40, 0x04)
    argEnd = conn.HexReadMemory(argEndAddress, 0x40, 0x04)
    envStart = conn.HexReadMemory(envStartAddress, 0x40, 0x04)
    envEnd = conn.HexReadMemory(envEndAddress, 0x40, 0x04)

    return pgdBase, codeStart, codeEnd, dataStart, dataEnd, brkStart, brkEnd, argStart, argEnd, envStart, envEnd

def GetPhysicalAddresses(conn, pgdBaseAddress, codeStartAddress, codeEndAddress, dataStartAddress, dataEndAddress, brkStartAddress, brkEndAddress, argStartAddress, argEndAddress, envStartAddress, envEndAddress):
    physCodeStartAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', codeStartAddress, pgdBaseAddress)
    physCodeEndAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', codeEndAddress, pgdBaseAddress)
    physDataStartAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', dataStartAddress, pgdBaseAddress)
    physDataEndAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', dataEndAddress, pgdBaseAddress)

    physbrkStartAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', brkStartAddress, pgdBaseAddress)
    physbrkEndAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', brkEndAddress, pgdBaseAddress)
    physargStartAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', argStartAddress, pgdBaseAddress)
    physargEndAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', argEndAddress, pgdBaseAddress)
    physenvStartAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', envStartAddress, pgdBaseAddress)
    physenvEndAddress = conn.TranslateVirtToPhys('BBB', '4.19.82-ti-rt-r31', envEndAddress, pgdBaseAddress)

    return physCodeStartAddress, physCodeEndAddress, physDataStartAddress, physDataEndAddress, physbrkStartAddress, physbrkEndAddress, physargStartAddress, physargEndAddress, physenvStartAddress, physenvEndAddress

def SaveProcessMemory(conn, directoryPath, PID, codeStartAddress, codeEndAddress, dataStartAddress, dataEndAddress, brkStartAddress, brkEndAddress, argStartAddress, argEndAddress, envStartAddress, envEndAddress, pgdBaseAddress):
    conn.NonIntrusiveSaveBinary(directoryPath + PID +'-code.bin ', codeStartAddress, codeEndAddress, pgdBaseAddress)
    conn.NonIntrusiveSaveBinary(directoryPath + PID + '-data.bin ', dataStartAddress, dataEndAddress, pgdBaseAddress, False)

    conn.NonIntrusiveSaveBinary(directoryPath + PID + '-brk.bin ', brkStartAddress, brkEndAddress, pgdBaseAddress, False)
    conn.NonIntrusiveSaveBinary(directoryPath + PID + '-arg.bin ', argStartAddress, argEndAddress, pgdBaseAddress, False)
    conn.NonIntrusiveSaveBinary(directoryPath + PID + '-env.bin ', envStartAddress, envEndAddress, pgdBaseAddress, False)

def GetProcessMemory(conn, address, PID, mmAddress):
    if PID != '0x4301':
        return

    pgdBaseAddress, codeStartAddress, codeEndAddress, dataStartAddress, dataEndAddress, brkStartAddress, brkEndAddress, argStartAddress, argEndAddress, envStartAddress, envEndAddress = GetProcessMemoryAddress(conn, int(mmAddress, 16))
    physCodeStartAddress, physCodeEndAddress, physDataStartAddress, physDataEndAddress, physbrkStartAddress, physbrkEndAddress, physargStartAddress, physargEndAddress, physenvStartAddress, physenvEndAddress = GetPhysicalAddresses(conn, pgdBaseAddress, codeStartAddress, codeEndAddress, dataStartAddress, dataEndAddress, brkStartAddress, brkEndAddress, argStartAddress, argEndAddress, envStartAddress, envEndAddress)

    print('\nPID %s (%s)' % (PID, mmAddress))
    print('VIRTUAL: \t Code Start : %s, Code End : %s, Data Start : %s, Data End : %s' % (codeStartAddress, codeEndAddress, dataStartAddress, dataEndAddress))
    print('PHYSICAL: \t Code Start : %s, Code End : %s, Data Start : %s, Data End : %s\n' % (physCodeStartAddress, physCodeEndAddress, physDataStartAddress, physDataEndAddress))
    SaveProcessMemory(conn, 'extracted_files/', PID, codeStartAddress, codeEndAddress, dataStartAddress, dataEndAddress, brkStartAddress, brkEndAddress, argStartAddress, argEndAddress, envStartAddress, envEndAddress, pgdBaseAddress)

def TraverseNextTaskStruct(sharedIndex, address):
    existingShared = shared_memory.SharedMemory(name='lauterbach2')
    sharedArray = np.ndarray((MAX_PROCESS_COUNT,), dtype=np.int64, buffer=existingShared.buf)

    nextAddress = address
    conn = Lauterbach()
    conn.Connect()

    while not nextAddress in sharedArray[:]:
        with sharedIndex.get_lock():
            sharedArray.put(sharedIndex.value, nextAddress)
            sharedIndex.value += 1

        mmAddress = GetMMStructAddress(conn, nextAddress)
        if mmAddress != '0x0':
            PID = GetPID(conn, nextAddress)
            GetProcessMemory(conn, nextAddress, PID, mmAddress)

        nextAddress = GetNextAddress(conn, nextAddress)
    conn.Disconnect()

def TraversePreviousTaskStruct(sharedIndex, address):
    existingShared = shared_memory.SharedMemory(name='lauterbach2')
    sharedArray = np.ndarray((MAX_PROCESS_COUNT,), dtype=np.int64, buffer=existingShared.buf)

    previousAddress = address
    conn = Lauterbach(True, '20001')
    conn.Connect()

    while not previousAddress in sharedArray[:]:
        with sharedIndex.get_lock():
            sharedArray.put(sharedIndex.value, previousAddress)
            sharedIndex.value += 1

        mmAddress = GetMMStructAddress(conn, previousAddress)
        if mmAddress != '0x0':
            PID = GetPID(conn, previousAddress)
            GetProcessMemory(conn, previousAddress, PID, mmAddress)

        previousAddress = GetPreviousAddress(conn, previousAddress)
    conn.Disconnect()

if __name__ == '__main__':
    mainConn = Lauterbach()
    mainConn.Connect()

    mmAddress = GetMMStructAddress(mainConn, INIT_TASK)
    if mmAddress != '0x0':
        InitPID = GetPID(mainConn, INIT_TASK)
        GetProcessMemory(mainConn, INIT_TASK, InitPID, mmAddress)

    initNext = GetNextAddress(mainConn, INIT_TASK)
    initPrevious = GetPreviousAddress(mainConn, INIT_TASK)
    mainConn.Disconnect()

    sharedIndex = Value('L', 0)
    sharedMemory = shared_memory.SharedMemory(name='lauterbach', create=True, size=262144)
    sharedArray = np.ndarray((MAX_PROCESS_COUNT,), dtype=np.int64, buffer=sharedMemory.buf)
    sharedArray.put(sharedIndex.value, INT_INIT_TASK)
    sharedIndex.value += 1

    nextProcess = multiprocessing.Process(target=TraverseNextTaskStruct, args=(sharedIndex, initNext, ))
    previousProcess = multiprocessing.Process(target=TraversePreviousTaskStruct, args=(sharedIndex, initPrevious, ))

    startTime = time.time()
    nextProcess.start()
    previousProcess.start()

    nextProcess.join()
    previousProcess.join()
    stopTime = time.time()
    print('Time elapsed: %s sec' % (stopTime - startTime))

    sharedMemory.close()
    sharedMemory.unlink()