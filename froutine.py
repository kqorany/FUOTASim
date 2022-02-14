#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 An updated version of LoRaSim 0.2.1 to simulate Fuota Routine
 author: Khaled Abdelfadeel khaled.abdelfadeel@mycit.ie
"""

"""
 SYNOPSIS:
   ./froutine.py.py <nodes> <randomseed>
 DESCRIPTION:
    nodes
        number of nodes to simulate
    randomseed
        random seed
 OUTPUT
    The result of every simulation run will be appended to a file named expX.dat,
    whereby X is the experiment number. The file contains a space separated table
    of values for nodes, collisions, transmissions and total energy spent. The
    data file can be easily plotted using e.g. gnuplot.
"""


import simpy
import random
import numpy as np
import math
import sys
import re
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import operator

# Turn on/off graphics
graphics = 0

# This is an array with measured values for sensitivity
#sf7 = np.array([7,-126.5,-124.25,-120.75])
#sf8 = np.array([8,-127.25,-126.75,-124.0])
#sf9 = np.array([9,-131.25,-128.25,-127.5])
#sf10 = np.array([10,-132.75,-130.25,-128.75])
#sf11 = np.array([11,-134.5,-132.75,-130])
#sf12 = np.array([12,-133.25,-132.25,-132.25])
##########################
#sf7 = np.array([7,-123,-120,-117.0])
#sf8 = np.array([8,-126,-123,-120.0])
#sf9 = np.array([9,-129,-126,-123.0])
#sf10 = np.array([10,-132,-129,-126.0])
#sf11 = np.array([11,-134.53,-131.52,-128.51])
#sf12 = np.array([12,-137,-134,-131.0])
###########################
#sensi = np.array([sf7,sf8,sf9,sf10,sf11,sf12])
###########################
# Supported LoRaWAN DataRates Senestivities
DRsSens = np.array([-137,-134,-132,-129,-126,-123,-120])
###########################

IS7 = np.array([1,-8,-9,-9,-9,-9])
IS8 = np.array([-11,1,-11,-12,-13,-13])
IS9 = np.array([-15,-13,1,-13,-14,-15])
IS10 = np.array([-19,-18,-17,1,-17,-18])
IS11 = np.array([-22,-22,-21,-20,1,-20])
IS12 = np.array([-25,-25,-25,-24,-23,1])
IsoThresholds = np.array([IS7,IS8,IS9,IS10,IS11,IS12])

# DataRates
# 0: SF = 12, BW = 125 kHz
# 1: SF = 11, BW = 125 kHz
# 2: SF = 10, BW = 125 kHz
# 3: SF =  9, BW = 125 kHz
# 4: SF =  8, BW = 125 kHz
# 5: SF =  7, BW = 125 kHz
# 6: SF =  7, BW = 250 KHz
DataRates = np.array([0,1,2,3,4,5])
SpreadFactors = np.array([12,11,10,9,8,7,7])
Bandwidths = np.array([125,125,125,125,125,125,250])
CodingRates = np.array([1,1,1,1,1,1,1])
TXPowers = np.array([0,2,4,6,8,10,12,14])

# last time the gateway acked a package
nearstACK1p = [0,0,0] # 3 channels with 1% duty cycle
nearstACK10p = 0 # one channel with 10% duty cycle
AckMessLen = 0
######################################################
# packet error model assumming independent Bernoulli
######################################################
def ber_reynders(eb_no, sf):
    """Given the energy per bit to noise ratio (in db), compute the bit error for the SF"""
    return norm.sf(math.log(sf, 12)/math.sqrt(2)*eb_no)

def ber_reynders_snr(snr, sf, bw, cr):
    """Compute the bit error given the SNR (db) and SF"""
    Temp = [4.0/5,4.0/6,4.0/7,4.0/8]
    CR = Temp[cr-1]
    BW = bw*1000.0
    eb_no =  snr - 10*math.log10(BW/2**sf) - 10*math.log10(sf) - 10*math.log10(CR) + 10*math.log10(BW)
    return ber_reynders(eb_no, sf)

def per(sf,bw,cr,rssi,pl):
    """Compute the packet error given the RSSI (dbB), DR, and Pcklen"""
    snr = rssi  +174 - 10*math.log10(bw) - 6
    return 1 - (1 - ber_reynders_snr(snr, sf, bw, cr))**(pl*8)
################################################
# pathloss model
################################################
def pathloss(dist):
    """Compute the rssi from two-component PL, given the distance"""
    if (dist < 0):
        print("ERROR: distance must be larger than 0")
    if (dist < 400):
        d0 = 92.67
        Lpld0 = 128.63
        gamma = 1.05
        sd = 8.72
    else:
        d0 = 37.27
        Lpld0 = 132.54
        gamma = 0.8
        sd = 3.34
    return (Lpld0 + 10*gamma*math.log10(dist/d0) + np.random.normal(-sd, sd))
################################################
# Transmission range
################################################
def TransRange(rssi, dist):
    """Compute the transmission range given the posision and rssi"""
    if (dist < 400):
        d0 = 92.67
        Lpld0 = 128.63
        gamma = 1.05
        sd = 8.72
    else:
        d0 = 37.27
        Lpld0 = 132.54
        gamma = 0.8
        sd = 3.34
    return (d0*(10**((rssi-Lpld0)/(10.0*gamma))))
##################################################################
# this function checks if the gateway can send a multicast,
# Fragmentation, clockcorrection, and startsession downlink commands
##################################################################
def checkcommand(node):
    global  nearstACK1p
    global  nearstACK10p
    # check ack in the first window
    chanlindex=[872000000, 864000000, 860000000].index(node.packet.freq)
    timeofacking = env.now + 1  # one sec after receiving the packet
    if not node.multicast:
        node.dtype = 'multicast'
        commandlen = 29 # McGroupSetupReq
    elif not node.fragmentation:
        commandlen = 10 # FragSessionSetupReq
        node.dtype = 'fragmentation'
    elif not node.sync:
        commandlen = 5 # AppTimeAns
        node.dtype = 'sync'
    elif not node.start:
        commandlen = 10 # McClassCSessionReq
        node.dtype = 'start'

    if (timeofacking >= nearstACK1p[chanlindex]):
        # the command can be sent
        print("gateway transmits downlink of type {0.dtype}".format(node))
        node.packet.downlink = True
        tempairtime = airtime(SpreadFactors[node.parameters.dr],CodingRates[node.parameters.dr],commandlen+LoRaWANMAChdr,Bandwidths[node.parameters.dr])
        nearstACK1p[chanlindex] = timeofacking+(tempairtime/0.01)
        nodes[node.packet.nodeid].rxtime += tempairtime
        return node.packet.downlink, node.parameters.dr, commandlen
    else:
        # the command can not be sent in the first window
        node.packet.downlink = False
        Tsym = (2.0**SpreadFactors[node.packet.dr])/(Bandwidths[node.packet.dr]*1000.0) # sec
        Tpream = (8 + 4.25)*Tsym
        nodes[node.packet.nodeid].rxtime += Tpream

    # chcek the second window
    timeofacking = env.now + 2  # two secs after receiving the packet
    if (timeofacking >= nearstACK10p):
        # the command can be sent
        print("gateway transmits downlink of type {0.dtype}".format(node))
        node.packet.acked = True
        tempairtime = airtime(12,CodingRates[node.parameters.dr],commandlen+LoRaWANMAChdr,Bandwidths[node.parameters.dr])
        nearstACK10p = timeofacking+(tempairtime/0.1)
        nodes[node.packet.nodeid].rxtime += tempairtime
        return node.packet.downlink, 0, commandlen
    else:
        # the command can not be sent in the second window either
        node.packet.downlink = False
        Tsym = (2.0**12)/(Bandwidths[node.packet.dr]*1000) # sec
        Tpream = (8 + 4.25)*Tsym
        nodes[node.packet.nodeid].rxtime += Tpream
        return node.packet.downlink, 0, 0
###########################################################
# check for collisions at base station
# Note: called before a packet (or rather node) is inserted into the list
############################################################
def checkcollision(packet):
    col = 0 # flag needed since there might be several collisions for packet
    processing = 0
    for i in range(0,len(packetsAtBS)):
        if packetsAtBS[i].packet.processed == 1:
            processing = processing + 1
    if (processing > maxBSReceives):
        print("too long:", len(packetsAtBS))
        packet.processed = 0
    else:
        packet.processed = 1

    if packetsAtBS:
        print("CHECK node {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(
             packet.nodeid, SpreadFactors[packet.dr], Bandwidths[packet.dr], packet.freq,
             len(packetsAtBS)))
        for other in packetsAtBS:
            if other.nodeid != packet.nodeid:
               print(">> node {} (sf:{} bw:{} freq:{:.6e})".format(
                   other.nodeid, SpreadFactors[other.packet.dr], Bandwidths[other.packet.dr], other.packet.freq))
               if(full_collision == 1 or full_collision == 2):
                   if frequencyCollision(packet, other.packet) \
                   and timingCollision(packet, other.packet):
                       # check who collides in the power domain
                       if (full_collision == 1):
                          # Capture effect
                          c = powerCollision_1(packet, other.packet)
                       else:
                          # Capture + Non-orthognalitiy SFs effects
                          c = powerCollision_2(packet, other.packet)
                       # mark all the collided packets
                       # either this one, the other one, or both
                       for p in c:
                          p.collided = 1
                          if p == packet:
                             col = 1
                   else:
                       # no freq or timing collision, all fine
                       pass
               else:
                   # simple collision
                   if frequencyCollision(packet, other.packet) \
                   and sfCollision(packet, other.packet):
                       packet.collided = 1
                       other.packet.collided = 1  # other also got lost, if it wasn't lost already
                       col = 1
        return col
    return 0
####################################################################
# frequencyCollision, conditions
#        |f1-f2| <= 120 kHz if f1 or f2 has bw 500
#        |f1-f2| <= 60 kHz if f1 or f2 has bw 250
#        |f1-f2| <= 30 kHz if f1 or f2 has bw 125
####################################################################
def frequencyCollision(p1,p2):
    if (abs(p1.freq-p2.freq)<=120 and (Bandwidths[p1.dr]==500 or p2.freq==500)):
        print("frequency coll 500")
        return True
    elif (abs(p1.freq-p2.freq)<=60 and (Bandwidths[p1.dr]==250 or p2.freq==250)):
        print("frequency coll 250")
        return True
    else:
        if (abs(p1.freq-p2.freq)<=30):
            print("frequency coll 125")
            return True
        #else:
    print("no frequency coll")
    return False

def sfCollision(p1, p2):
    if SpreadFactors[p1.dr] == SpreadFactors[p2.dr]:
        print("collision sf node {} and node {}".format(p1.nodeid, p2.nodeid))
        # p2 may have been lost too, will be marked by other checks
        return True
    print("no sf collision")
    return False
################################################################
# check only the capture between the same spreading factor
################################################################
def powerCollision_1(p1, p2):
    #powerThreshold = 6
    print("pwr: node {0.nodeid} {0.rssi:3.2f} dBm node {1.nodeid} {1.rssi:3.2f} dBm; diff {2:3.2f} dBm".format(p1, p2, round(p1.rssi - p2.rssi,2)))
    if SpreadFactors[p1.dr] == SpreadFactors[p2.dr]:
       if abs(p1.rssi - p2.rssi) < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
            print("collision pwr both node {} and node {}".format(p1.nodeid, p2.nodeid))
            # packets are too close to each other, both collide
            # return both pack ets as casualties
            return (p1, p2)
       elif p1.rssi - p2.rssi < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
            # p2 overpowered p1, return p1 as casualty
            print("collision pwr node {} overpowered node {}".format(p2.nodeid, p1.nodeid))
            return (p1,)
       print("p1 wins, p2 lost")
       # p2 was the weaker packet, return it as a casualty
       return (p2,)
    else:
       return ()
#################################################################
# check the capture effect and imperfect-orthogonality of SFs
#################################################################
def powerCollision_2(p1, p2):
    #powerThreshold = 6
    print("DR: node {0.nodeid} {0.dr} node {1.nodeid} {1.dr}".format(p1, p2))
    print("pwr: node {0.nodeid} {0.rssi:3.2f} dBm node {1.nodeid} {1.rssi:3.2f} dBm; diff {2:3.2f} dBm".format(p1, p2, round(p1.rssi - p2.rssi,2)))
    if SpreadFactors[p1.dr] == SpreadFactors[p2.dr]:
       if abs(p1.rssi - p2.rssi) < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
           print("collision pwr both node {} and node {}".format(p1.nodeid, p2.nodeid))
           # packets are too close to each other, both collide
           # return both packets as casualties
           return (p1, p2)
       elif p1.rssi - p2.rssi < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
           # p2 overpowered p1, return p1 as casualty
           print("collision pwr node {} overpowered node {}".format(p2.nodeid, p1.nodeid))
           print("capture - p2 wins, p1 lost")
           return (p1,)
       print("capture - p1 wins, p2 lost")
       # p2 was the weaker packet, return it as a casualty
       return (p2,)
    else:
       if p1.rssi-p2.rssi > IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
          print("P1 is OK")
          if p2.rssi-p1.rssi > IsoThresholds[SpreadFactors[p2.dr]-7][SpreadFactors[p1.dr]-7]:
              print("p2 is OK")
              return ()
          else:
              print("p2 is lost")
              return (p2,)
       else:
           print("p1 is lost")
           if p2.rssi-p1.rssi > IsoThresholds[SpreadFactors[p2.dr]-7][SpreadFactors[p1.dr]-7]:
               print("p2 is OK")
               return (p1,)
           else:
               print("p2 is lost")
               return (p1,p2)

#################################################################
# Check time collision
#################################################################
def timingCollision(p1, p2):
    # assuming p1 is the freshly arrived packet and this is the last check
    # we've already determined that p1 is a weak packet, so the only
    # way we can win is by being late enough (only the first n - 5 preamble symbols overlap)
    # assuming 8 preamble symbols
    Npream = 8
    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = (2**SpreadFactors[p1.dr]/(Bandwidths[p1.dr]*1.0))*(Npream - 5) # to sec
    # check whether p2 ends in p1's critical section
    p2_end = p2.addtime + p2.rectime
    p1_cs = env.now + (Tpreamb/1000.0)
    print("collision timing node {} ({},{},{}) node {} ({},{})".format(
        p1.nodeid, env.now - env.now, p1_cs - env.now, p1.rectime,
        p2.nodeid, p2.addtime - env.now, p2_end - env.now
    ))
    if p1_cs < p2_end:
        # p1 collided with p2 and lost
        print("not late enough")
        return True
    print("saved by the preamble")
    return False
##################################################################
# this function computes the airtime of a packet
# according to LoraDesignGuide_STD.pdf
##################################################################
def airtime(sf,cr,pl,bw):
    H = 0        # implicit header disabled (H=0) or not (H=1)
    DE = 0       # low data rate optimization enabled (=1) or not (=0)
    Npream = 8   # number of preamble symbol (12.25  from Utz paper)

    if bw == 125 and sf in [11, 12]:
        # low data rate optimization mandated for BW125 with SF11 and SF12
        DE = 1
    if sf == 6:
        # can only have implicit header with SF6
        H = 1

    Tsym = (2.0**sf)/bw  # msec
    Tpream = (Npream + 4.25)*Tsym
    print("sf", sf, " cr", cr, "pl", pl, "bw", bw)
    payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
    Tpayload = payloadSymbNB * Tsym
    return ((Tpream + Tpayload)/1000.0)  # to secs
######################################################################
# this function creates a gateway
######################################################################
class myGateway ():
    def __init__(self, gatewayid):
        self.gatewayid = gatewayid
        self.gl = 8   # antenna gains and losses
        self.dr = 12
        self.txpow = 14
        # gateway placement
        self.x = maxDist+5
        self.y = maxDist+5
        # gateway stats
        self.txtime  = 0
        self.rxtime  = 0
######################################################################
# this function creates a node
######################################################################
class myNode():
    def __init__(self, nodeid, bs):
        self.nodeid = nodeid
        self.bs = bs
        self.gl = 2.2  # antenna gains and losses
        self.first = 1
        self.x = 0
        self.y = 0
        self.utype = enumerate(['reg','ack','sync'])
        self.dtype = enumerate(['multicast','fragmentation','sync','start'])
        # node stats
        self.multicast = False
        self.fragmentation = False
        self.sync = False
        self.start = False
        self.commandack = False
        # counters
        self.usent = 0     # uplink sent
        self.ucoll = 0     # uplink collisions
        self.uerror = 0    # uplink error
        self.ulost = 0     # uplink lost
        self.dsent = 0     # downlink sent
        self.nodown = 0    # can't downlink
        self.dlost = 0     # downlink lost
        self.derror = 0    # downlink error
        self.starttime = 0 # minimum time to start
        # radio times
        self.rxtime = 0
        self.txtime = 0
        self.idealtime = 0
        # this is very complex prodecure for placing nodes
        # and ensure minimum distance between each pair of nodes
        found = 0
        rounds = 0
        global nodes
        while (found == 0 and rounds < 100):
            a = random.random()
            b = random.random()
            if b<a:
                a,b = b,a
            posx = b*maxDist*math.cos(2*math.pi*a/b)+gateway.x
            posy = b*maxDist*math.sin(2*math.pi*a/b)+gateway.y
            if len(nodes) > 0:
                for index, n in enumerate(nodes):
                    dist = np.sqrt(((abs(n.x-posx))**2)+((abs(n.y-posy))**2))
                    if dist >= 10:
                        found = 1
                        self.x = posx
                        self.y = posy
                    else:
                        rounds = rounds + 1
                        if rounds == 100:
                            print("could not place new node, giving up")
                            exit(-1)
            else:
                print("first node")
                self.x = posx
                self.y = posy
                found = 1
        self.dist = np.sqrt((self.x-gateway.x)*(self.x-gateway.x)+(self.y-gateway.y)*(self.y-gateway.y))
        print('node %d' %nodeid, "x", self.x, "y", self.y, "dist: ", self.dist)

        # graphics for node
        global graphics
        if (graphics == 1):
            global ax
            ax.add_artist(plt.Circle((self.x, self.y), 2, fill=True, color='blue'))
#######################################################################
# assign datarate according to distance, mimiking ADR algorithm
#######################################################################
class nodeParameters():
    def __init__(self, nodeid, distance, antenna):
        self.nodeid = nodeid
        self.txpow = 14
        self.dr = 0  # SF12, BW125K, CR1
        self.freq = random.choice([872000000, 864000000, 860000000])
        # simulate the ADR
        Lpls = []
        for i in range(0,10):
            Lpls.append(pathloss(distance))
        Lpl = np.mean(Lpls)
        print("Lpl:", Lpl)
        Prx = self.txpow - Lpl
        minairtime = 9999
        maxdr = 0
        print("Prx:", Prx)
        for i in DataRates:  # DRs [0,1,2,3,4,5] BW=125KHz
            if ((DRsSens[i]) < Prx):
                maxdr = i
        print("best DR", maxdr, "best sf:", SpreadFactors[maxdr], " best bw: ", Bandwidths[maxdr])
        # balance the distribution
        global drDistribution, txDistribution
        newdr = maxdr
        ratios = [0.06,0.08,0.08,0.11,0.22,0.45]
        if float((drDistribution[newdr]+1))/nrNodes >= ratios[newdr]:
            for i in range(maxdr-1,-1,-1):
                if float((drDistribution[i]+1))/nrNodes <= ratios[i]:
                    newdr = i
                    break
        print("new DR", newdr, "new sf:", SpreadFactors[newdr], " new bw: ", Bandwidths[newdr])
        self.dr = newdr
        drDistribution[self.dr]+=1;
        txDistribution[int(self.txpow)-2]+=1;
#############################################################
# this function creates an uplink packet (associated with a node)
# it also sets all parameters, currently random
#############################################################
class ulPacket():
    def __init__(self, nodeid, freq, dr, txpow, pl, antenna):
        self.nodeid = nodeid
        self.freq = freq
        self.dr = dr
        self.txpow = txpow
        self.pl = pl
        self.symtime = (2.0**SpreadFactors[self.dr])/(Bandwidths[self.dr]*1000) # sec
        self.arrivetime = 0
        self.addtime = 0
        self.transrange = TransRange(self.txpow + antenna - DRsSens[dr], maxDist)
        self.rectime = airtime(SpreadFactors[self.dr],CodingRates[self.dr],LoRaWANMAChdr+self.pl,Bandwidths[self.dr])
        print("nodeid: ", self.nodeid, "upacket symtime: ", self.symtime, "upacket rectime: ", self.rectime)
        # packet stats
        self.processed = False
        self.collided = False
        self.lost = False
        self.error = False
        self.downlink = True
        self.dlost = False
        self.derror = False
####################################################################
# main discrete event loop, runs for each node
# a global list of packet being processed at the gateway
# is maintained
####################################################################
def transmit(env,node):
    while not(node.sync and node.multicast and node.fragmentation and node.start):
        # node still in the FUOTA routine
        node.packet.rssi = node.parameters.txpow + node.gl - pathloss(node.dist)

        if node.commandack:
            node.utype = 'ack'
            if (node.dtype == 'multicast'):
                node.packet.pl = 1 # McGroupStatusAns
            elif (node.dtype == 'fragmentation'):
                node.packet.pl = 1 # FragSessionSetupAns
            elif (node.dtype == 'start'):
                node.packet.pl = 4 # McClassCSessionAns
        elif not(node.multicast and node.fragmentation):
            node.utype = 'reg'
            node.packet.pl = 15 # regular app packet
        elif not node.sync:
            node.utype = 'sync'
            node.packet.pl = 5 # AppTimeReq
        elif not node.start:
            node.utype = 'reg'
            node.packet.pl = 15 # regular app packet
        # update the rectime
        node.packet.rectime = airtime(SpreadFactors[node.parameters.dr],CodingRates[node.parameters.dr],LoRaWANMAChdr+node.packet.pl,Bandwidths[node.parameters.dr])
        # this is the first message ever
        if (node.first == 1):
            node.first = 0
            yield env.timeout(random.uniform(0,1000)/1000.0) #  1 secs
        else:
            yield env.timeout((99*node.packet.rectime)+(random.uniform(0,500)/1000.0)) # 0.5 sec randomization

        print("node {0.nodeid} transmits uplink of type {0.utype}".format(node))

        # time sending and receiving
        # packet arrives -> add to base station
        node.usent = node.usent + 1
        if (node in packetsAtBS):
            print("ERROR: packet already in")
        else:
            if node.packet.rssi < DRsSens[node.parameters.dr]:
                print("node {}: packet will be lost".format(node.nodeid))
                node.packet.lost = True
            else:
                node.packet.lost = False
                if (per(SpreadFactors[node.packet.dr],Bandwidths[node.packet.dr],CodingRates[node.packet.dr],node.packet.rssi,node.packet.pl) < random.uniform(0,1)):
                    # OK CRC
                    node.packet.error = False
                else:
                    # Bad CRC
                    node.packet.error = True
                # adding packet if no collision
                if (checkcollision(node.packet)==1):
                    node.packet.collided = True
                else:
                    node.packet.collided = False

                packetsAtBS.append(node)
                node.packet.addTime = env.now

        yield env.timeout(node.packet.rectime)

        if (node.packet.lost == False\
                and node.packet.error == False\
                and node.packet.collided == False):

            if (node.utype == 'ack'):
                # downlink command is acked fine
                node.commandack = False
                if (node.dtype == 'multicast'):
                    node.multicast = True
                elif (node.dtype == 'fragmentation'):
                    node.fragmentation = True
                elif (node.dtype == 'start'):
                    node.start = True

            else: # ack or sync
                results = checkcommand(node) # (falg, dr, pl)
                if(results[0]):
                    node.dsent = node.dsent + 1
                    node.packet.downlink = True
                    # there is a downlink command
                    # check if the downlink is lost or not
                    gatewayrssi = gateway.txpow + gateway.gl - pathloss(node.dist)
                    if(gatewayrssi > DRsSens[results[1]]):
                        # the command is not lost
                        node.packet.dlost = False
                        if (per(SpreadFactors[results[1]],Bandwidths[results[1]],CodingRates[results[1]],gatewayrssi,LoRaWANMAChdr+results[2]) < random.uniform(0,1)):
                            # OK CRC
                            node.packet.derror = False
                            if (node.dtype == 'sync'):
                                node.commandack = False
                                node.sync = True
                                node.starttime = env.now
                            else:
                                node.commandack = True # the command needs ack
                        else:
                            # Bad CRC
                            node.packet.derror = True
                    else:
                        # command is lost
                        node.packet.dlost = True
                else:
                    node.packet.downlink = False

        if node.packet.lost:
            node.ulost = node.ulost + 1
        elif node.packet.error:
            node.uerror = node.uerror + 1
        elif node.packet.collided:
            node.ucoll = node.ucoll + 1
        elif not node.packet.downlink:
            node.nodown = node.nodown + 1
        elif node.packet.dlost:
            node.dlost = node.dlost + 1
        elif node.packet.derror:
            node.derror = node.derror + 1

        # complete packet has been received by base station
        # can remove it
        if (node in packetsAtBS):
            packetsAtBS.remove(node)
        # reset the packet
        node.packet.collided = False
        node.packet.processed = False
        node.packet.lost = False
        node.packet.error = False
        node.packet.downlink = True
        node.packet.dlost = False
        node.packet.derror = False

####################################################
# "main" program
####################################################

# get arguments
if len(sys.argv) >= 3:
    nrNodes = int(sys.argv[1])
    full_collision = 2
    rndmdSeed  = int(sys.argv[2])
    print("Nodes:", nrNodes)
    print("Full Collision: ", full_collision)
    print("Random Seed: ", rndmdSeed)
else:
    print("usage: ./froutine.py <nodes> <randomseed>")
    exit(-1)

# intiate the random module
random.seed(rndmdSeed)

# global stuff
nodes = []
nodeder1 = [0 for i in range(0,nrNodes)]
nodeder2 = [0 for i in range(0,nrNodes)]
tempdists = [0 for i in range(0,nrNodes)]
packetsAtBS = []
drDistribution = [0 for x in range(0,6)]
txDistribution = [0 for x in range(0,13)]
env = simpy.Environment()

# lorawan mac header
LoRaWANMAChdr = 8
# maximum number of packets the BS can receive at the same time
maxBSReceives = 8

# max distance: 300m in city, 3000 m outside (5 km Utz experiment)
# also more unit-disc like according to Utz
bsId = 1

# global stats
nrUplinks = 0
nrCollisions = 0
nrDownlinks = 0
nrUlost = 0
nrUerror = 0
nrNoDown = 0
nrDlost = 0
nrDerror = 0
minstarttime = 0

# calculate the maximum distance from the gateway
Lpl =  2 - np.amin(DRsSens)
maxDist = 37.27*(10**((Lpl-132.54)/(10.0*0.8)))
print("maxDist:", maxDist)  # 1322.38 meters in this case


# setup the gateway to transmit the fragments
gateway = myGateway(bsId)
for i in range(0,nrNodes):
    node = myNode(i,bsId)
    nodes.append(node)
    node.parameters = nodeParameters(node.nodeid, node.dist, node.gl)
    node.packet = ulPacket(node.nodeid, node.parameters.freq, node.parameters.dr, node.parameters.txpow, 15, node.gl)
    env.process(transmit(env,node))

# start simulation
env.run()

#prepare show
if (graphics == 1):
    plt.ion()
    plt.figure()
    ax = plt.gcf().gca()
    # XXX should be base station position
    ax.add_artist(plt.Circle((gateway.x, gateway.y), 3, fill=True, color='green'))
    ax.add_artist(plt.Circle((gateway.x, gateway.y), maxDist, fill=False, color='green'))
    plt.xlim([0, gateway.x + maxDist + 5])
    plt.ylim([0, gateway.y + maxDist + 5])
    plt.draw()
    plt.show()

# global stats
nrUplinks = sum(n.usent for n in nodes)
nrCollisions = sum(n.ucoll for n in nodes)
nrDownlinks = sum(n.dsent for n in nodes)
nrUlost = sum(n.ulost for n in nodes)
nrUerror = sum(n.uerror for n in nodes)
nrNoDown = sum(n.nodown for n in nodes)
nrDlost = sum(n.dlost for n in nodes)
nrDerror = sum(n.derror for n in nodes)
minstarttime = sum(env.now-n.starttime for n in nodes)/nrNodes

# compute energy
# Transmit consumption in mA from -2 to +17 dBm
TX = [22, 22, 22, 23,                                      # RFO/PA0: -2..1
      24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
      82, 85, 90,                                          # PA_BOOST/PA1: 15..17
      105, 115, 125]                                       # PA_BOOST/PA1+PA2: 18..20
RX = 16         # mA
IDEAL = 0.006   # mA
V = 3.0     # voltage XXX
for i in range(0,nrNodes):
        nodes[i].idealtime = env.now - (nodes[i].rxtime+nodes[i].txtime)

networkEnergy = float(sum(((node.txtime * TX[int(node.parameters.txpow)+2])+(node.rxtime * RX)+(node.idealtime * IDEAL)) * V  for node in nodes)  / 1e3)

# data extraction rate
der = (nrUplinks-(nrCollisions+nrUlost+nrUerror))/float(nrUplinks) if nrUplinks!=0 else 0

# data extraction rate per node
#for i in range(0,nrNodes):
#    tempdists[i] = nodes[i].dist
#    nodeder1[i] = ((nodes[i].sent-nodes[i].coll)/(float(nodes[i].sent)) if float(nodes[i].sent)!=0 else 0)
#    nodeder2[i] = (nodes[i].recv/(float(nodes[i].sent)) if float(nodes[i].sent)!=0 else 0)
# calculate the fairness indexes per node
#nodefair1 = (sum(nodeder1)**2/(nrNodes*sum([i*float(j) for i,j in zip(nodeder1,nodeder1)])) if (sum([i*float(j) for i,j in zip(nodeder1,nodeder1)]))!=0 else 0)
#nodefair2 = (sum(nodeder2)**2/(nrNodes*sum([i*float(j) for i,j in zip(nodeder2,nodeder2)])) if (sum([i*float(j) for i,j in zip(nodeder2,nodeder2)]))!=0 else 0)

print("=============================")
print("           RESULTS           ")
print("=============================")
print("Nodes: ", nrNodes)
print("maxDist: ", maxDist)
print("Random Seed: ", rndmdSeed)
print("total energy (in J): ", networkEnergy)
print("der: ", der)
print("time: ", env.now)
print("uplink sent: ", nrUplinks)
print("downlink sent: ", nrDownlinks)
print("uplink collisions: ", nrCollisions)
print("uplink lost: ", nrUlost)
print("uplink bad CRC: ", nrUerror)
print("no downlink: ", nrNoDown)
print("downlink lost: ", nrDlost)
print("downlink error: ", nrDerror)
print("=============================")
print("rdDdistribution: ", drDistribution)
print("txDistribution: ", txDistribution)
print("=============================")
print("MinStartTime: ", minstarttime)

# save experiment data into a dat file that can be read by e.g. gnuplot
# name of file would be:  exp0.dat for experiment 0
fname = str("froutine") + ".dat"
print(fname)
if os.path.isfile(fname):
     res= "\n" + str(rndmdSeed) + ", " + str(full_collision) + ", " + str(nrNodes) + ", " + str(nrUplinks) + ", "  + str(nrCollisions) + ", "  + str(nrUlost) + ", "  + str(nrUerror) + ", " + str(nrDownlinks) + ", " +str(nrNoDown) + ", " +str(nrDlost) + ", " +str(nrDerror)+ ", " + str(env.now)+ ", " + str(minstarttime)+ ", " + str(der)+ ", " + str(networkEnergy)+ ", "  + str(drDistribution)
else:
     res = "#randomseed, collType, nrNodes, nrUplinks, nrCollisions, nrUlost, nrUerror, nrDownlinks, nrNoDown, nrDlost, nrDerror, Time, MinStartTime, DER, networkEnergy, DR0, DR1, DR2, DR3, DR4, DR5\n" + str(rndmdSeed) + ", " + str(full_collision) + ", " + str(nrNodes) + ", " + str(nrUplinks) + ", "  + str(nrCollisions) + ", "  + str(nrUlost) + ", "  + str(nrUerror) + ", " + str(nrDownlinks) + ", " +str(nrNoDown) + ", " +str(nrDlost) + ", " +str(nrDerror)+ ", " + str(env.now)+ ", " + str(minstarttime)+ ", " + str(der)+ ", " + str(networkEnergy)+ ", "  + str(drDistribution)
newres=re.sub('[^#a-zA-Z0-9 \n\.]','',res)
print(newres)
with open(fname, "a") as myfile:
    myfile.write(newres)
myfile.close()

# this can be done to keep graphics visible
if (graphics == 1):
    raw_input('Press Enter to continue ...')

# with open('nodes.txt','w') as nfile:
#     for n in nodes:
#         nfile.write("{} {} {}\n".format(n.x, n.y, n.nodeid))
# with open('basestation.txt', 'w') as bfile:
#     bfile.write("{} {} {}\n".format(bsx, bsy, 0)
