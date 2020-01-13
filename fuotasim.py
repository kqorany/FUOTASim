#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 A Fuota process simulatorfor Class C multicast
 author: Khaled Abdelfadeel khaled.abdelfadeel@ieee.org
"""

"""
 SYNOPSIS:
   ./fuotasim.py <nodes> <datarate> <multicast> <periodicity> <firmwaresize> <fragsize> <redundent> <randomseed>
 DESCRIPTION:
    nodes
        number of nodes to simulate
    daatrate
        data rate used to send the firmware
    multicast
        class for the multicast
    periodicity
        Ping Periodicity of Class B
    firmwaresize
        size of firmware to be sent in bytes
    fragsize
        size of a fragment in bytes
    redundent
        number of redundent fragments
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
        print "ERROR: distance must be larger than 0"
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
        print "too long:", len(packetsAtBS)
        packet.processed = 0
    else:
        packet.processed = 1

    if packetsAtBS:
        print "CHECK node {} (sf:{} bw:{} freq:{:.6e}) others: {}".format(
             packet.nodeid, SpreadFactors[packet.dr], Bandwidths[packet.dr], packet.freq,
             len(packetsAtBS))
        for other in packetsAtBS:
            if other.nodeid != packet.nodeid:
               print ">> node {} (sf:{} bw:{} freq:{:.6e})".format(
                   other.nodeid, SpreadFactors[other.packet.dr], Bandwidths[other.packet.dr], other.packet.freq)
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
        print "frequency coll 500"
        return True
    elif (abs(p1.freq-p2.freq)<=60 and (Bandwidths[p1.dr]==250 or p2.freq==250)):
        print "frequency coll 250"
        return True
    else:
        if (abs(p1.freq-p2.freq)<=30):
            print "frequency coll 125"
            return True
        #else:
    print "no frequency coll"
    return False

def sfCollision(p1, p2):
    if SpreadFactors[p1.dr] == SpreadFactors[p2.dr]:
        print "collision sf node {} and node {}".format(p1.nodeid, p2.nodeid)
        # p2 may have been lost too, will be marked by other checks
        return True
    print "no sf collision"
    return False
################################################################
# check only the capture between the same spreading factor
################################################################
def powerCollision_1(p1, p2):
    #powerThreshold = 6
    print "pwr: node {0.nodeid} {0.rssi:3.2f} dBm node {1.nodeid} {1.rssi:3.2f} dBm; diff {2:3.2f} dBm".format(p1, p2, round(p1.rssi - p2.rssi,2))
    if SpreadFactors[p1.dr] == SpreadFactors[p2.dr]:
       if abs(p1.rssi - p2.rssi) < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
            print "collision pwr both node {} and node {}".format(p1.nodeid, p2.nodeid)
            # packets are too close to each other, both collide
            # return both pack ets as casualties
            return (p1, p2)
       elif p1.rssi - p2.rssi < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
            # p2 overpowered p1, return p1 as casualty
            print "collision pwr node {} overpowered node {}".format(p2.nodeid, p1.nodeid)
            return (p1,)
       print "p1 wins, p2 lost"
       # p2 was the weaker packet, return it as a casualty
       return (p2,)
    else:
       return ()
#################################################################
# check the capture effect and imperfect-orthogonality of SFs
#################################################################
def powerCollision_2(p1, p2):
    #powerThreshold = 6
    print "DR: node {0.nodeid} {0.dr} node {1.nodeid} {1.dr}".format(p1, p2)
    print "pwr: node {0.nodeid} {0.rssi:3.2f} dBm node {1.nodeid} {1.rssi:3.2f} dBm; diff {2:3.2f} dBm".format(p1, p2, round(p1.rssi - p2.rssi,2))
    if SpreadFactors[p1.dr] == SpreadFactors[p2.dr]:
       if abs(p1.rssi - p2.rssi) < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
           print "collision pwr both node {} and node {}".format(p1.nodeid, p2.nodeid)
           # packets are too close to each other, both collide
           # return both packets as casualties
           return (p1, p2)
       elif p1.rssi - p2.rssi < IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
           # p2 overpowered p1, return p1 as casualty
           print "collision pwr node {} overpowered node {}".format(p2.nodeid, p1.nodeid)
           print "capture - p2 wins, p1 lost"
           return (p1,)
       print "capture - p1 wins, p2 lost"
       # p2 was the weaker packet, return it as a casualty
       return (p2,)
    else:
       if p1.rssi-p2.rssi > IsoThresholds[SpreadFactors[p1.dr]-7][SpreadFactors[p2.dr]-7]:
          print "P1 is OK"
          if p2.rssi-p1.rssi > IsoThresholds[SpreadFactors[p2.dr]-7][SpreadFactors[p1.dr]-7]:
              print "p2 is OK"
              return ()
          else:
              print "p2 is lost"
              return (p2,)
       else:
           print "p1 is lost"
           if p2.rssi-p1.rssi > IsoThresholds[SpreadFactors[p2.dr]-7][SpreadFactors[p1.dr]-7]:
               print "p2 is OK"
               return (p1,)
           else:
               print "p2 is lost"
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
    Tpreamb = 2**SpreadFactors[p1.dr]/(1.0*Bandwidths[p1.dr]) * (Npream - 5)

    # check whether p2 ends in p1's critical section
    p2_end = p2.addtime + p2.rectime
    p1_cs = env.now + (Tpreamb/1000.0)  # to sec
    print "collision timing node {} ({},{},{}) node {} ({},{})".format(
        p1.nodeid, env.now - env.now, p1_cs - env.now, p1.rectime,
        p2.nodeid, p2.addtime - env.now, p2_end - env.now
    )
    if p1_cs < p2_end:
        # p1 collided with p2 and lost
        print "not late enough"
        return True
    print "saved by the preamble"
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
    print "sf", sf, " cr", cr, "pl", pl, "bw", bw
    payloadSymbNB = 8 + max(math.ceil((8.0*pl-4.0*sf+28+16-20*H)/(4.0*(sf-2*DE)))*(cr+4),0)
    Tpayload = payloadSymbNB * Tsym
    return ((Tpream + Tpayload)/1000.0)  # to secs
######################################################################
# this function creates a gateway
######################################################################
class myGateway ():
    def __init__(self, gatewayid, datarate, multicast, periodicity, frmwrsize, fragsize, redundants):
        self.gatewayid = gatewayid
        self.gl = 8   # antenna gains and losses
        self.dr = datarate
        self.txpow = 14
        self.firmwaresize = frmwrsize
        self.fragsize = fragsize
        self.redundants = redundants
        self.freq = 869525000
        # gateway placement
        self.x = maxDist+5
        self.y = maxDist+5
        # calculate the number of fragments
        temp = math.ceil(float(frmwrsize)/(fragsize-2)) # 2 for the frag header
        self.nrfrags = temp + redundants
        global minFragsFull
        minFragsFull =  temp + 3
        # gateway stats
        self.txtime  = 0
        self.rxtime  = 0
        # multicast class
        self.multicast = multicast
        # for class B
        self.beacons = 0 # number of beacons have be sent so far
        self.slotoffset = 0
        self.periodicity = periodicity # [0-7]
        self.nrping = 2**(7-self.periodicity)
        self.beaconperiod = 128 # sec
        self.slotperiod = 0.96 * 2**self.periodicity # sec
        self.slotlength = 0.03 # sec
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
        # node stats
        self.rxfrags = 0
        self.updatedfull = False
        self.synced = False  # for class b beacons
        self.fraglost = 0
        self.fragerror = 0
        self.rxtime = 0
        self.txtime = 0
        self.idealtime = 0
        self.updatetime = 0
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
                            print "could not place new node, giving up"
                            exit(-1)
            else:
                print "first node"
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
        print "Lpl:", Lpl
        Prx = self.txpow - Lpl
        minairtime = 9999
        maxdr = 0
        print "Prx:", Prx
        for i in DataRates:  # DRs [0,1,2,3,4,5] BW=125KHz
            if ((DRsSens[i]) < Prx):
                maxdr = i
        print "best DR", maxdr, "best sf:", SpreadFactors[maxdr], " best bw: ", Bandwidths[maxdr]
        # balance the distribution
        global drDistribution, txDistribution
        newdr = maxdr
        ratios = [0.06,0.08,0.08,0.11,0.22,0.45]
        if float((drDistribution[newdr]+1))/nrNodes >= ratios[newdr]:
            for i in range(maxdr-1,-1,-1):
                if float((drDistribution[i]+1))/nrNodes <= ratios[i]:
                    newdr = i
                    break
        print "new DR", newdr, "new sf:", SpreadFactors[newdr], " new bw: ", Bandwidths[newdr]
        self.dr = newdr
        drDistribution[self.dr]+=1;
        txDistribution[int(self.txpow)-2]+=1;
#############################################################
# this function creates a downlink packet (associated with the gateway)
# it also sets all parameters, currently random
#############################################################
class dlPacket():
    def __init__(self, gatewayid, freq, dr, txpow, pl, antenna):
        self.gatewayid = gatewayid
        self.freq = freq
        self.dr = dr
        self.txpow = txpow
        self.pl = pl
        self.symtime = (2.0**SpreadFactors[self.dr])/(Bandwidths[self.dr]*1000) # sec
        self.arrivetime = 0
        self.addtime = 0
        self.transrange = TransRange(self.txpow + antenna - DRsSens[dr], maxDist)
        self.rectime = airtime(SpreadFactors[self.dr],CodingRates[self.dr],LoRaWANMACHdr+self.pl,Bandwidths[self.dr])
        print "gatewayid: ", self.gatewayid, "dlpacket symtime: ", self.symtime, "dlpacket rectime: ", self.rectime
        # packet stats
        self.processed = 0
#############################################################
# this function creates a beacon (associated with the gateway)
#############################################################
class beaconParameters():
    def __init__(self, gatewayid, datarate):
        self.gatewayid = gatewayid
        self.freq = 869525000 # regional specs
        self.dr = datarate
        self.txpow = 27  # max at this freq band
        self.pl = 17     # RFU(2)|Time(4)|CRC(2)|GwSpecific(7)|CRC(2)
        self.symtime = (2.0**SpreadFactors[self.dr])/(Bandwidths[self.dr]*1000) # sec
        self.arrivetime = 0
        self.addtime = 0
        self.transrange = TransRange(self.txpow - DRsSens[self.dr], maxDist)
        self.rectime = airtime(SpreadFactors[self.dr],CodingRates[self.dr],LoRaWANMACHdr+self.pl,Bandwidths[self.dr])
        print "gatewayid: ", self.gatewayid, "beacon symtime: ", self.symtime, "beacon rectime: ", self.rectime
#############################################################
# gateway transmits
#############################################################
def dltransmit(env,gateway):
    global nextBeacon
    while gateway.nrfrags > 0.0:
        if gateway.multicast == 1:
            if env.now >= nextBeacon:
                # a new beacon just has been sent
                gateway.slotoffset = random.choice(range(0,gateway.nrping+1))
                timeAfterRandom = gateway.beacon.addtime + gateway.beacon.rectime+(gateway.slotoffset*gateway.slotlength)
                if (env.now < timeAfterRandom):
                    yield env.timeout(timeAfterRandom - env.now)
        # wait the airtime of a fragment
        gateway.packet.addtime= env.now
        yield env.timeout(gateway.packet.rectime)

        gateway.txtime += gateway.packet.rectime
        gateway.nrfrags -= 1
        print "gateway {0.gatewayid} nrFragments {0.nrfrags}".format(gateway)

        # check the reciption at every node
        for i in range(0,nrNodes):
            if nodes[i].updatedfull == False:
                #[TODO] node synchronization should be checked here
                rssiAtNode = gateway.txpow + gateway.gl - pathloss(nodes[i].dist)
                # check if the rssi > DR senestivity
                if rssiAtNode >= DRsSens[gateway.dr]:
                    # check the packet error
                    if (per(SpreadFactors[gateway.dr],Bandwidths[gateway.dr],CodingRates[gateway.dr],rssiAtNode,gateway.packet.pl) < random.uniform(0,1)):
                        # Fragment can be received ok
                        nodes[i].rxfrags += 1
                        # check if enough fragments have been recveived
                        if nodes[i].rxfrags >= minFragsFull:
                            nodes[i].updatedfull = True
                            nodes[i].updatetime = env.now
                    else:
                        # Fragment will be received but corrupted
                        nodes[i].fragerror += 1
                    # RX time class B
                    if gateway.multicast == 1:
                        # class B - receive the whole fragment
                        nodes[i].rxtime += gateway.packet.rectime
                else:
                    nodes[i].fraglost += 1
                    print "node {}: will loss this fragment".format(nodes[i].nodeid)
                    if gateway.multicast == 1:
                        # class B - receive the preamble part
                        nodes[i].rxtime += 0.03  # 30 ms window
                # RX time
                if gateway.multicast == 0:
                    # class c - always in rx window
                    nodes[i].rxtime = env.now

        gateway.packet.processed = True

        # don't yield when the queue is empty already
        if gateway.nrfrags > 0.0:
            if multicast == 0:
                # class C - next fragment to obey 10% duty cycle
                yield env.timeout(9*gateway.packet.rectime)
            elif multicast == 1:
                # class B - next slot depends
                global emptypings
                emptypings = math.floor((9*gateway.packet.rectime)/(gateway.slotperiod))
                # update the rx time
                for i in range(0,nrNodes):
                     nodes[i].rxtime += emptypings*0.03
                yield env.timeout((emptypings+1)*gateway.slotperiod)
        # update the global stats
        if gateway.packet.processed == True:
            global nrProcessed
            nrProcessed += 1

        # reset the dlPacket
        gateway.packet.processed = False
#############################################################
# gateway transmits beacons
#############################################################
def btransmit(env,gateway):
    while gateway.nrfrags > 0.0:
        gateway.beacon.addtime = env.now
        # wait for the airtime of a beacon
        yield env.timeout(gateway.beacon.rectime)
        # check the reciption at every node
        for i in range(0,nrNodes):
            if nodes[i].updatedfull == False:
                rssiAtNode = gateway.beacon.txpow - pathloss(nodes[i].dist)
                # check if the rssi > DR senestivity
                if rssiAtNode >= DRsSens[gateway.dr]:
                    # check the packet error
                    if (per(SpreadFactors[gateway.dr],Bandwidths[gateway.dr],CodingRates[gateway.dr],rssiAtNode,gateway.beacon.pl) < random.uniform(0,1)):
                        # beacon can be received ok
                        nodes[i].synced = True
                    else:
                        # beacon will be received but corrupted
                        nodes[i].synced = False
                        print "node {}: will be out of synched".format(nodes[i].nodeid)
                else:
                    nodes[i].synced = False
                    print "node {}: will be out of synched".format(nodes[i].nodeid)
                # devices will always try to recive this
                nodes[i].rxtime += gateway.beacon.rectime

        gateway.beacons += 1
        # don't yield when the queue is empty already
        if gateway.nrfrags > 0.0:
            # next beacon
            global nextBeacon
            nextBeacon = env.now + gateway.beaconperiod
            yield env.timeout(gateway.beaconperiod)
###########################################################
# MAIN PROGRWM
###########################################################
if len(sys.argv) >= 9:
# get arguments
    nrNodes = int(sys.argv[1])
    dataRate = int(sys.argv[2])
    multicast = int(sys.argv[3])
    periodicity = int(sys.argv[4])
    frmwrSize = int(sys.argv[5])
    fragSize = int(sys.argv[6])
    nrRedundant = int(sys.argv[7])
    rndmdSeed  = int(sys.argv[8])
    print "Nodes: ", nrNodes
    print "DataRate: ", dataRate
    if multicast == 0:
        print "Multicast: Class C"
    elif multicast == 1:
        print "Multicast: Class B"
    elif multicast == 2:
        print "Multicast: Class X"
    print "Ping Periodicity: ", periodicity
    print "Firmware Size: ", frmwrSize
    print "Fragment Size: ", fragSize
    print "Redundant: ", nrRedundant
    print "Random Seed: ", rndmdSeed
else:
    print "usage: ./fuotasim.py <nodes> <datarate> <multicast> <periodicity> <firmwaresize> <fragsize> <redundent> <randomseed>"
    exit(-1)

# intiate the random module
random.seed(rndmdSeed)

# global stuff
nodes = []
dlpackets = []
updateEfficiency = 0
updateTime = 0
chlUtilization = 0

drDistribution = [0 for x in DataRates]
txDistribution = [0 for x in range(0,13)]
env = simpy.Environment()

# base station id
bsId = 1
# maximum number of packets the BS can receive at the same time
maxBSReceives = 8
# minimum number of fragments to be received for full update
minFragsFull = 0
# LoRaWAN MAC Header
LoRaWANMACHdr = 8
# global network stats
nrProcessed = 0
nrFraglost = 0
nrFragerror = 0
# global
nextBeacon = env.now
emptypings = 0

# calculate the maximum distance from the gateway
Lpl = 2 - np.amin(DRsSens)
maxDist = 37.27*(10**((Lpl-132.54)/(10.0*0.8)))
print "maxDist:", maxDist  # 1322.38 meters in this case

# setup the gateway to transmit the fragments
gateway = myGateway(bsId, dataRate, multicast, periodicity, frmwrSize, fragSize, nrRedundant)
gateway.packet = dlPacket(bsId, gateway.freq, gateway.dr, gateway.txpow, gateway.fragsize, gateway.gl)
if multicast == 1: # class B
    # assign a beacon class to the gateway
    gateway.beacon = beaconParameters(bsId, gateway.dr)

# devices receive
for i in range(0,nrNodes):
    # myNode takes period (in ms), base station id packetlen (in Bytes)
    node = myNode(i,bsId)
    nodes.append(node)
    node.parameters = nodeParameters(node.nodeid, node.dist, node.gl)
if multicast == 1: # class B
    env.process(btransmit(env,gateway))
# start the dltransmit process
env.process(dltransmit(env,gateway))


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

# start simulation
env.run()

# compute energy
# Transmit consumption in mA from -2 to +17 dBm
TX = [22, 22, 22, 23,                                      # RFO/PA0: -2..1
      24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
      82, 85, 90,                                          # PA_BOOST/PA1: 15..17
      105, 115, 125]                                       # PA_BOOST/PA1+PA2: 18..20
RX = 16         # mA
IDEAL = 0.006   # mA
V = 3.0         # voltage XXX

# Total Frag Lost
nrFraglost = sum(node.fraglost for node in nodes)
# Total Frag Corrupted
nrFragerror = sum(node.fragerror for node in nodes)
# Network Energy
if multicast == 1:
# class B
    for i in range(0,nrNodes):
        nodes[i].idealtime = env.now - nodes[i].rxtime

networkEnergy = float(sum(((node.txtime * TX[int(node.parameters.txpow)+2])+(node.rxtime * RX)+(node.idealtime * IDEAL)) * V  for node in nodes)  / 1e3)
# Total Update Time from all devices
updateTime = float(sum(node.updatetime for node in nodes))
# Update Efficiency
nrUpdated = float(sum(node.updatedfull==True for node in nodes))
updateEfficiency = nrUpdated/nrNodes
# Channel Efficiency
effeAirtime = airtime(SpreadFactors[gateway.dr], CodingRates[gateway.dr], fragSize-2, Bandwidths[gateway.dr]) # 2 for frag header
chlUtilization = (effeAirtime*math.ceil(float(frmwrSize)/(fragSize-2.0)))/env.now

print "============================="
print "           RESULTS           "
print "============================="
print "Nodes: ", nrNodes
print "DataRate: ", dataRate
if multicast == 0:
    print "Multicast: Class C"
elif multicast == 1:
    print "Multicast: Class B"
elif multicast == 2:
    print "Multicast: Class X"
print "Ping Periodicity: ", periodicity
print "Empty Pings: ", emptypings
print "Firmware Size: ", frmwrSize
print "Fragment Size: ", fragSize
print "Redundant: ", nrRedundant
print "Random Seed: ", rndmdSeed
print "============================="
print "Total fragments: ", math.ceil(float(frmwrSize)/(fragSize-2))+nrRedundant
print "sent fragments: ", nrProcessed
print "Minimum nrFrags for Update: ", minFragsFull
print "Total lost frags: ", nrFraglost
print "Total corrupted frags: ", nrFragerror
print "Total Energy (in J): ", networkEnergy
print "Update Time (in sec): ", updateTime
print "Update Efficiency: ", updateEfficiency
print "Channel Utilization: ", chlUtilization
print "============================="
print "rdDdistribution: ", drDistribution
print "txDistribution: ", txDistribution
print "Update Time: ", env.now
print "============================="

# save experiment data into a dat file that can be read by e.g. gnuplot
# name of file would be:  exp0.dat for experiment 0
fname = str("fuotasim") + ".dat"
print fname
if os.path.isfile(fname):
     res= "\n" + str(rndmdSeed) + ", " + str(nrNodes) + ", " + str(dataRate) + ", " + str(multicast) + ", " + str(periodicity) + ", " + str(emptypings) + ", " + str(frmwrSize) + ", "  + str(fragSize) + ", "  + str(nrRedundant) + ", "  + str(math.ceil(float(frmwrSize)/(fragSize-2))) + ", "  + str(nrProcessed) + ", " + str(nrFraglost) + ", " + str(nrFragerror) + ", " + str(networkEnergy) + ", " +str(updateTime) + ", " + str(updateEfficiency) + ", " + str(chlUtilization) + ", " + str(drDistribution)
else:
     res = "randomseed, nrNodes, DataRate, Multicast, Periodicity, EmptyPings, FirmSize, FragSize, nrRedundant, nrFrag, nrProcessed, nrFraglost, nrFragerror, networkEnergy, updateTime, updateEfficiency, chlUtilization, DR0, DR1, DR2, DR3, DR4, DR5\n" + str(rndmdSeed) + ", " + str(nrNodes) + ", " + str(dataRate) + ", " + str(multicast) + ", " + str(periodicity) + ", " + str(emptypings) + ", " + str(frmwrSize) + ", "  + str(fragSize) + ", "  + str(nrRedundant) + ", "  + str(math.ceil(float(frmwrSize)/(fragSize-2))) + ", "  + str(nrProcessed) + ", " + str(nrFraglost) + ", " + str(nrFragerror) + ", " +str(networkEnergy) + ", " +str(updateTime) + ", " + str(updateEfficiency) + ", " + str(chlUtilization) + ", " + str(drDistribution)
newres=re.sub('[^#a-zA-Z0-9 \n\.]','',res)
print newres
with open(fname, "a") as myfile:
    myfile.write(newres)
myfile.close()

# this can be done to keep graphics visible
if (graphics == 1):
    raw_input('Press Enter to continue ...')
