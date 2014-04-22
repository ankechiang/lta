from __future__ import division
import pylab
import numpy
import collections
import math, random
import datetime
import re
import os, optparse, sys, operator, itertools, json
from os import listdir
from os.path import isfile, join


verbose = False

def plot_curve(data, xlabel, NAMES, figname):
    FORMAT = ['r+']
    CNT, XS = [], []
    print data
    numbers = numpy.array(data)
    CNT.append(collections.Counter(numbers))
    XS.append(numpy.array(sorted(CNT[0].keys())))

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    ps = []

    sum_count = []
    sum_count.append(sum(CNT[0].values()))
    sum_all = 0
    for x in XS[0]:
        sum_all = sum_all + x*CNT[0][x]
    print "avg. numTrip of a user", sum_all*1.0/sum_count[0]
    print "avg. numTrip of a user", numpy.average(numpy.array(data))
    print "max. numTrip of a user", max(XS[0])
    print "min. numTrip of a user", min(XS[0])
    print "std. numTrip of a user", numpy.std(numpy.array(data))
    print "med. numTrip of a user", numpy.median(numpy.array(data))

    freq = open(figname+"."+str(0), 'w')
    for x in XS[0]:
        freq.write(str(x)+"\t"+str(CNT[0][x])+"\t"+str(CNT[0][x]*1.0/sum_count[0])+"\n")
    freq.close()

    ps.append(ax.plot(XS[0], [CNT[0][x]*1.0/sum_count[0] for x in XS[0]], FORMAT[0], mew=1, markersize=5, label = NAMES))
    ax.set_yscale('log')
    ax.set_xscale('log')
    pylab.xlabel(xlabel, fontsize=20)
    pylab.ylabel('Pr[X=x]', fontsize=20)
    ax.legend(loc='upper right', prop={'size':18})
    pylab.savefig(figname+'.png')



def roundTime(dt=None, roundTo=60):
    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    newdt = dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)
    return str(newdt.hour)+":"+str(newdt.minute)

def compute_round_time(date, time):
    d = date.split("/")
    d = map(int, d)
    t = time.split(":")
    return roundTime(datetime.datetime(d[2],d[1],d[0],int(t[0]),int(t[1]),0), roundTo=60*30)


'''
c, d, t, l, m, r, dt, tt

all     6114487 ,e:13:18:3787:BUS:980:5.70:14.450,x:13:18:980:BUS:980:5.70:14.450,e:15:23:3159:BUS:912:5.90:20.050,x:15:23:912:BUS:912:5.90:20.050
'''

def read_profile(profile):
    user = []
    for item in profile.split(",")[1:]:
        ele = item.split(":")
        user.append(tuple(item.split(":")))
    return user



def group_by_time(data_entry):
    data=sorted(data_entry,key=operator.itemgetter(0))

    distribution = {}
    for key,group in itertools.groupby(data,operator.itemgetter(0)):
        subdata = [(l,mm) for time, l, mm in list(group)]
        distribution[key] = subdata
    return distribution


def group_by_loc(intervals):
    data=sorted(intervals,key=operator.itemgetter(0))

    distribution = {}
    for key,group in itertools.groupby(data,operator.itemgetter(0)):
        subdata = [stay for loc, lat, lon, stay in list(group)]
        distribution[key] = []
        distribution[key].append(numpy.average(numpy.array(subdata)))
        distribution[key].append(numpy.std(numpy.array(subdata)))
        distribution[key].append(numpy.median(numpy.array(subdata)))
        distribution[key].append(len(subdata))
    return distribution


def group_by_segment(intervals):
    data=sorted(intervals,key=operator.itemgetter(0,1))

    distribution = {}
    for key_src,group in itertools.groupby(data,operator.itemgetter(0)):
        subdata = [(dest, duration) for src, dest, duration in list(group)]
        freq = len(subdata)*1.0
        for key_dest, seg in itertools.groupby(subdata,operator.itemgetter(0)):
            segdata = [duration for dest, duration in list(seg)]
            distribution[key_src+":"+key_dest] = []
            distribution[key_src+":"+key_dest].append(numpy.average(numpy.array(segdata)))
            distribution[key_src+":"+key_dest].append(numpy.std(numpy.array(segdata)))
            distribution[key_src+":"+key_dest].append(numpy.median(numpy.array(segdata)))
            distribution[key_src+":"+key_dest].append(len(segdata))
            distribution[key_src+":"+key_dest].append(len(segdata)*1.0/freq)
    return distribution


def write_points(points, submission_file=None):
    writer = open(submission_file, 'w')
    for lon, lat, hh, mm in points:
        writer.write(str(lon)+"\t"+str(lat)+"\t"+str(hh)+"\t"+str(mm)+"\n")
    writer.close()


def write_sessions(user, sessions, submission_file=None):
    writer = open(submission_file, 'w')
    for l, stay, begin, end, duration in sessions:
        writer.write(l+"\t"+str(stay)+"\t"+begin+"\t"+end+"\t"+str(duration)+"\n")
    writer.close()

def get_station_name(rtsmap, converter, lid):
    lname = "NA"
    if not lid == "-99":
        if lid in converter.keys(): lid = converter[lid]
        if lid in rtsmap.keys(): lname = rtsmap[lid][2]
        else: lname = "NA"
    else: lname = "-99"
    return lname

def write_reviews(rtsmap, converter, reviews, submission_file=None):
    writer = open(submission_file+".csv", 'w')
    writer.write("date"+","+"departure"+","+"arrival"+","+"duration"+","+"dwell station"+","+"dwell time"+"\n")
    pdate, pdepart, parrival, pdur = "", "", "", ""
    for l, stay, begin, end, duration in reviews:
        loc = ""
        b,e = begin.split(":"), end.split(":")
        loc = get_station_name(rtsmap, converter, loc)
        loc_depart, loc_arrival = "", ""
        if pdepart and parrival:
            loc_depart = get_station_name(rtsmap, converter, pdepart.split(":")[2])
            loc_arrival = get_station_name(rtsmap, converter, parrival.split(":")[2])
        if loc == "-99":
            writer.write(pdate+","+pdepart+","+parrival+","+loc_depart+","+loc_arrival+","+str(duration)+","+loc+","+str(stay)+"\n")
        else:
            writer.write(pdate+","+pdepart+","+parrival+","+loc_depart+","+loc_arrival+","+str(pdur)+","+loc_arrival+","+str(stay)+"\n")
        pdate, pdepart, parrival, pdur = b[0], b[1]+":"+b[2]+":"+b[3], e[1]+":"+e[2]+":"+e[3], duration
    writer.close()

def write_intervals(intervals, submission_file=None):
    writer = open(submission_file, 'w')
    for lat, lon, hh1, mm1, hh2, mm2 in intervals:
        writer.write(str(lat)+"\t"+str(lon)+"\t"+str(hh1)+"\t"+str(mm1)+"\t"+str(hh2)+"\t"+str(mm2)+"\n")
    writer.close()


def write_nodes(distributions, submission_file=None):
    writer = open(submission_file, 'w')
    for loc, val in distributions.iteritems():
        writer.write(str(loc)+"\t"+str(val[0])+"\t"+str(val[1])+"\t"+str(val[2])+"\t"+str(val[3])+"\n")
    writer.close()


def write_edges(distributions, submission_file=None):
    writer = open(submission_file, 'w')
    for loc, val in distributions.iteritems():
        writer.write(str(loc.replace(":", "\t"))+"\t"+str(val[0])+"\t"+str(val[1])+"\t"+str(val[2])+"\t"+str(val[3])+"\t"+str(val[4])+"\n")
    writer.close()


def write_count(sessions_cnt, submission_file=None):
    writer = open(submission_file, 'w')
    for cnt in sessions_cnt: writer.write(str(cnt)+"\n")
    writer.close()


def write_actions(actions, submission_file=None):
    writer = open(submission_file, 'w')
    for action, hh, mm in actions:
        writer.write(str(action)+"\t"+str(hh)+"\t"+str(mm)+"\n")
    writer.close()


# CREATE (PennyM:Person {name:'Penny Marshall', born:1943})
# CREATE (TomH)-[:ACTED_IN {roles:['Jimmy Dugan']}]->(ALeagueofTheirOwn)
def write_graphs(user, graph, submission_file=None):
    writer = open(submission_file, 'w')
    nodes, edges = [], []
    for begin, end, duration, gis_begin, gis_end in graph:
        # Nodes
        if begin not in nodes:
            writer.write("CREATE ("+gis_begin[2].replace(" ", "")+begin+":Station"+user+" {name:'"+gis_begin[2]+"', lon:'"+gis_begin[0]+"', lat:'"+gis_begin[1]+"'})\n")
            nodes.append(begin)
        if end not in nodes:
            writer.write("CREATE ("+gis_end[2].replace(" ", "")+end+":Station"+user+" {name:'"+gis_end[2]+"', lon:'"+gis_end[0]+"', lat:'"+gis_end[1]+"'})\n")
            nodes.append(end)

        # Edges
        if (begin, end) in edges: continue
        writer.write("CREATE ("+gis_begin[2].replace(" ", "")+begin+")-[:TO {travel_cost:"+duration+"}]->("+gis_end[2].replace(" ", "")+end+")\n")
        edges.append((begin, end))
    writer.close()


def convert_station_to_gis(distribution, gismap):
    XS = []
    for key, value in distribution.iteritems():
        for s, mm in value:
            if str(s) not in gismap.keys(): continue
            XS.append((gismap[str(s)][1], gismap[str(s)][0], key, mm))
    return XS


def read_gismap(gis_path):
    gisfile = open(gis_path, "r")
    gismap = {}
    for p in gisfile:
        tokens = p.strip().split("\t")
        if tokens[0] not in gismap.keys():
            gismap[tokens[0]] = []
            gismap[tokens[0]].append(tokens[2])
            gismap[tokens[0]].append(tokens[1])
    return gismap


def read_rts_gismap(gis_path):
    gisfile = open(gis_path, "r")
    gismap, converter = {}, {}
    for p in gisfile:
        key = ""
        tokens = p.strip().split("\t")
        if len(tokens) < 3: continue
        if len(tokens) > 4: converter[tokens[0]] = tokens[4]
        key = tokens[0]
        if key not in gismap.keys():
            gismap[key] = []
            gismap[key].append(tokens[3])
            gismap[key].append(tokens[2])
            gismap[key].append(tokens[1])
    print converter
    return gismap, converter


def time_difference(dd1, hh1, mm1, dd2, hh2, mm2):
    dt1 = datetime.datetime(2014,3,int(dd1),int(hh1),int(mm1),0)
    dt2 = datetime.datetime(2014,3,int(dd2),int(hh2),int(mm2),0)

    diff = dt2 - dt1
    return divmod(diff.days * 86400 + diff.seconds, 60)

def check_date(dd):
    w = datetime.date(2012,1,int(dd)).weekday()
    if w == 5 or w == 6:
        return False
    return True

def time_addition(dd1, hh1, mm1, mm_delta):
    delta = datetime.timedelta(minutes = mm_delta)
    if verbose: print dd1, hh1, mm1, datetime.datetime(2014,3,int(dd1),int(hh1),int(mm1),0) + delta
    return (datetime.datetime(2014,3,int(dd1),int(hh1),int(mm1),0) + delta).strftime("%Y-%m-%d %H:%M:%S")

def gen_train_data(gismap, traj_path, train_path, test_path, thr_ets):
    onlyfiles = [ f for f in listdir(traj_path) if isfile(join(traj_path,f)) ]
    cnt = 1
    for f in onlyfiles:
        #trajfile = open(traj_path+"/trajectory.Jan01_08.txt", "r")
        trajfile = open(traj_path+"/"+f, "r")
        print "generating training data from", f
        for line in trajfile:
            tokens = line.strip().split("\t")
            user, events = tokens[1], tokens[2]
            all_events = read_profile(events)
            if len(all_events) < thr_ets or len(all_events) > 20000:  continue

            entries = [(int(hh), int(l), int(mm))for c, d, hh, mm, l, m, r, dt, tt in all_events]
            distribution = group_by_time(entries)
            #if len(distribution) > 10:
            train_data = convert_station_to_gis(distribution, gismap)
            split_pts = math.ceil(len(train_data)*0.8)
            write_points(train_data[:int(split_pts)], train_path+"/"+user)
            write_points(train_data[int(split_pts):], test_path+"/"+user)
            cnt+=1
    print "generate", cnt, "training data"


def segmentation(all_events, threshold):
    segments = []
    begin, end = "", ""
    ccx, ddx, hhx, mmx, llx = "", "", "", "", ""
    cce, dde, hhe, mme, lle = "", "", "", "", ""
    for c, d, hh, mm, l, m, r, dt, tt in all_events:
        if verbose: print c, d, hh, mm, "l=", l
        if c == "x":
            ccx, ddx, hhx, mmx, llx = c, d, hh, mm, l
        if c == "e":
            if lle == llx:
                # skip invalid trip
                #if not lle == "":
                #    if verbose: print "\t", hhe+":"+mme+":"+lle, "-->", hhx+":"+mmx+":"+llx
                #    segments.append(("?", "?", begin, end, "?"))
                cce, dde, hhe, mme, lle = c, d, hh, mm, l
                begin = dde+":"+hhe+":"+mme+":"+lle
            else:
                # output valid trip
                if not begin == "":
                    B = begin.split(":")
                    if end == "":
                        stay = -99
                        last_stop = "-99"
                    else:
                        X = end.split(":")
                        last_stop = X[3]
                        (stay, tmp) = time_difference(X[0],X[1],X[2], B[0],B[1],B[2])
                    end = ddx+":"+hhx+":"+mmx+":"+llx
                    (travel, tmp) = time_difference(B[0],B[1],B[2], ddx, hhx, mmx)
                    if check_date(B[0]):
                        segments.append((last_stop, stay, begin, end, travel))
                    if verbose: print "\t", str((stay/60))+"(hr)@"+last_stop, begin, end, str(travel)+"(min)"

                # update lle, llx, begin, end
                cce, dde, hhe, mme, lle = c, d, hh, mm, l
                begin = dde+":"+hhe+":"+mme+":"+lle
    return segments


def count_session_data(traj_path, tripCnt_dist_path, thr_time):
    onlyfiles = [ f for f in listdir(traj_path) if isfile(join(traj_path,f)) ]
    cnt = 1
    session_cnt = []
    for f in onlyfiles:
        trajfile = open(traj_path+"/"+f, "r")
        print "generating session data from", f
        for line in trajfile:
            tokens = line.strip().split("\t")
            user, events = tokens[1], tokens[2]
            all_events = read_profile(events)

            segments = segmentation(all_events, thr_time)
            session_cnt.append(len(segments))
            cnt+=1
            #print len(segments)
    write_count(session_cnt, tripCnt_dist_path)
    print "count", cnt, "session data"
    return session_cnt


def gen_session_data(rtsmap, converter, traj_path, train_session_path, test_session_path, train_review_path, thr_time, thr_ets):
    if not os.path.exists(train_review_path): os.makedirs(train_review_path)
    onlyfiles = [ f for f in listdir(traj_path) if isfile(join(traj_path,f)) ]
    cnt = 1
    for f in onlyfiles:
        trajfile = open(traj_path+"/"+f, "r")
        print "generating session data from", f
        for line in trajfile:
            tokens = line.strip().split("\t")
            user, events = tokens[1], tokens[2]
            all_events = read_profile(events)
            if len(all_events) < thr_ets or len(all_events) > 40000:  continue

            #entries = [(int(hh), int(l), int(mm))for c, d, hh, mm, l, m, r, dt, tt in all_events]
            segments = segmentation(all_events, thr_time)
            #train_data = convert_station_to_gis(segments, gismap)
            split_pts = math.ceil(len(segments)*0.8)
            write_sessions(user, segments[:int(split_pts)], train_session_path+"/"+user)
            write_sessions(user, segments[int(split_pts):], test_session_path+"/"+user)
            write_reviews(rtsmap, converter, segments[:int(split_pts)], train_review_path+"/"+user)
            cnt+=1
    print "generate", cnt, "session data"



def gen_graph_data(rtsmap, converter, train_session_path, train_graph_path):
    onlyfiles = [ f for f in listdir(train_session_path) if isfile(join(train_session_path,f)) ]
    #onlyfiles = ['6581804']
    cnt = 1
    for f in onlyfiles:
        sessionfile = open(train_session_path+"/"+f, "r")
        print "generating graph data from", f
        segments = []
        for line in sessionfile:
            tokens = line.strip().split("\t")
            if len(tokens) < 4: print len(tokens), line.strip()
            begin, end, duration = tokens[2].split(":")[3], tokens[3].split(":")[3], tokens[4]
            if begin in converter.keys(): begin = converter[begin]
            if end in converter.keys(): end = converter[end]
            if begin not in rtsmap.keys() or end not in rtsmap.keys(): continue
            gis_begin, gis_end = rtsmap[begin], rtsmap[end]
            segments.append((begin, end, duration, gis_begin, gis_end))
            cnt+=1
        write_graphs(f, segments, train_graph_path+"/"+f)
    print "generate", cnt, "graph data"


def sampling(dd1, hh1, mm1, interval, cell_size):
    samples = []
    num = math.ceil(interval/cell_size)
    for i in xrange(int(num)):
        delta = (i+1)*cell_size
        if delta < interval:
            #print (i+1)*cell_size, time_addition(dd1,hh1,mm1,delta)
            samples.append(time_addition(dd1,hh1,mm1,delta))
    return samples

def gen_psudoss_data(rtsmap, converter, spRatio, train_session_path, train_psudoss_path, test_psudoss_path, train_path):
    for sp in spRatio:
        if not os.path.exists(train_psudoss_path+"/"+str(sp)): os.makedirs(train_psudoss_path+"/"+str(sp))

    onlyfiles = [ f for f in listdir(train_session_path) if isfile(join(train_session_path,f)) ]
    #onlyfiles = ['6581804']
    pts, psudopts = [], []
    cnt = 1
    for f in onlyfiles:
        sessionfile = open(train_session_path+"/"+f, "r")
        print "generating psudo data for", f
        last_dd, last_hh, last_mm = "", "", ""
        pts, psudopts = [], {}
        for line in sessionfile:
            tokens = line.strip().split("\t")
            if len(tokens) < 4: print len(tokens), line.strip()
            last_loc, stay, begin, end, duration = tokens[0], float(tokens[1]), tokens[2], tokens[3], float(tokens[4])
            dd_b, hh_b, mm_b, loc_b = begin.split(":")
            dd_e, hh_e, mm_e, loc_e = end.split(":")

            # original gis
            if loc_b in converter.keys(): loc_b = converter[loc_b]
            if loc_e in converter.keys(): loc_e = converter[loc_e]
            if loc_b not in rtsmap.keys() or loc_e not in rtsmap.keys(): continue
            gis_begin, gis_end = rtsmap[loc_b], rtsmap[loc_e]

            if verbose: print tokens
            # sampling/psudo sampling
            samples, psudosamples = [], []
            if stay > 0 and last_dd and last_hh and last_mm:
                for sp in spRatio:
                    psudopts[sp] = []
                    psudosamples = sampling(last_dd, last_hh, last_mm, stay, sp)
                    for sample in psudosamples:
                        t = sample.split(" ")[1].split(":")
                        psudopts[sp].append((last_gis[1], last_gis[0], t[0], t[1]))
                        if verbose: print "\t", last_loc, last_gis, sample


            pts.append((gis_begin[1], gis_begin[0], hh_b, mm_b))
            if verbose: print "---", loc_b, gis_begin, hh_b, mm_b
            for sp in spRatio:
                if sp not in psudopts.keys(): psudopts[sp] = []
                samples = sampling(dd_b, hh_b, mm_b, duration, sp)
                for sample in samples:
                    date, tmp = sample.split(" ")[0].split("-"), sample.split(" ")[1].split(":")
                    (gp1,a) = time_difference(dd_b, hh_b, mm_b, date[2], tmp[0], tmp[1])
                    (gp2,b) = time_difference(date[2], tmp[0], tmp[1], dd_e, hh_e, mm_e)
                    if gp1 < gp2:
                        psudopts[sp].append((gis_begin[1], gis_begin[0], tmp[0], tmp[1]))
                        if verbose: print "\t", loc_b, gis_begin, sample, "("+str(gp1)+"<"+str(gp2)+")"
                    else:
                        psudopts[sp].append((gis_end[1], gis_end[0], tmp[0], tmp[1]))
                        if verbose: print "\t", loc_e, gis_end, sample, "("+str(gp1)+">"+str(gp2)+")"
            pts.append((gis_end[1], gis_end[0], hh_e, mm_e))
            if verbose: print "===", loc_e, gis_end, hh_e, mm_e
            last_dd, last_hh, last_mm, last_loc, last_gis = dd_e, hh_e, mm_e, loc_b, gis_end
            cnt+=1

        random.shuffle(pts)
        split_pts = math.ceil(len(pts)*0.8)
        if int(split_pts) == 0 or len(pts)==0 : continue
        for sp in spRatio:
            write_points(psudopts[sp] + pts[:int(split_pts)], train_psudoss_path+"/"+str(sp)+"/"+f)
        write_points(pts[int(split_pts):], test_psudoss_path+"/"+f)
        write_points(pts[:int(split_pts)], train_path+"/"+f)
    print "generate", cnt, "psudo session data"


def gen_action_data(train_session_path, train_action_path, test_action_path):
    onlyfiles = [ f for f in listdir(train_session_path) if isfile(join(train_session_path,f)) ]
    #onlyfiles = ['6581804']
    pts = []
    cnt = 1
    for f in onlyfiles:
        sessionfile = open(train_session_path+"/"+f, "r")
        print "generating action data from", f
        pts = []
        #try:
        for line in sessionfile:
            tokens = line.strip().split("\t")
            if len(tokens) < 4: print len(tokens), line.strip()
            begin, end = tokens[2], tokens[3]
            dd_b, hh_b, mm_b, loc_b = begin.split(":")
            dd_e, hh_e, mm_e, loc_e = end.split(":")

            if verbose: print tokens

            pts.append((1.0, hh_b, mm_b))
            pts.append((0, hh_e, mm_e))
            cnt+=1

        random.shuffle(pts)
        split_pts = math.ceil(len(pts)*0.8)
        if len(pts) == 0: continue
        write_actions(pts[:int(split_pts)], train_action_path+"/"+f)
        write_actions(pts[int(split_pts):], test_action_path+"/"+f)
        #except:
        #    print f
        #    pass
    print "generate", cnt, "action data"


def gen_node_data(rtsmap, converter, train_session_path, train_node_path, thr_dist):
    if not os.path.exists(train_node_path): os.makedirs(train_node_path)

    onlyfiles = [ f for f in listdir(train_session_path) if isfile(join(train_session_path,f)) ]
    #onlyfiles = ['6581804']
    pts, intervals = [], []
    cnt = 1
    for f in onlyfiles:
        sessionfile = open(train_session_path+"/"+f, "r")
        print "generating node attributes for", f
        last_dd, last_hh, last_mm, last_gis = "", "", "", ""
        pts, intervals = [], []
        for line in sessionfile:
            tokens = line.strip().split("\t")
            if len(tokens) < 4: print len(tokens), line.strip()
            last_loc, stay, begin, end, duration = tokens[0], float(tokens[1]), tokens[2], tokens[3], float(tokens[4])
            dd_b, hh_b, mm_b, loc_b = begin.split(":")
            dd_e, hh_e, mm_e, loc_e = end.split(":")

            # original gis
            if loc_b in converter.keys(): loc_b = converter[loc_b]
            if loc_e in converter.keys(): loc_e = converter[loc_e]
            if last_loc in converter.keys(): last_loc = converter[last_loc]
            if loc_b not in rtsmap.keys() or loc_e not in rtsmap.keys(): continue
            gis_begin, gis_end = rtsmap[loc_b], rtsmap[loc_e]

            if verbose: print tokens
            if stay > 0 and last_gis and last_hh and last_mm:
                if last_loc == loc_b:
                    if verbose: print "\t",last_loc,str(last_hh)+":"+str(last_mm),str(hh_b)+":"+str(mm_b),stay
                    intervals.append((last_loc, last_gis[1], last_gis[0], stay))
                else:
                    gis = []
                    gis.append([float(last_gis[1]), float(last_gis[0])])
                    gis.append([float(gis_begin[1]), float(gis_begin[0])])
                    tmp = numpy.array(gis)
                    if numpy.linalg.norm(tmp[0] - tmp[1]) < thr_dist:
                        intervals.append((last_loc, last_gis[1], last_gis[0], stay))
                        if verbose: print "\t",str(last_loc)+"->"+str(loc_b),numpy.linalg.norm(tmp[0] - tmp[1]), gis[0], gis[1]

            last_dd, last_hh, last_mm, last_loc, last_gis = dd_e, hh_e, mm_e, loc_b, gis_end
            cnt+=1
        if intervals:
            distributions = group_by_loc(intervals)
            write_nodes(distributions, train_node_path+"/"+f)
    print "generate", cnt, "node attribute data"



def gen_edge_data(rtsmap, converter, train_session_path, train_edge_path):
    if not os.path.exists(train_edge_path): os.makedirs(train_edge_path)

    onlyfiles = [ f for f in listdir(train_session_path) if isfile(join(train_session_path,f)) ]
    #onlyfiles = ['6581804']
    pts, intervals = [], []
    cnt = 1
    for f in onlyfiles:
        sessionfile = open(train_session_path+"/"+f, "r")
        print "generating edge attributes for", f
        last_dd, last_hh, last_mm, last_gis = "", "", "", ""
        pts, intervals = [], []
        for line in sessionfile:
            tokens = line.strip().split("\t")
            if len(tokens) < 4: print len(tokens), line.strip()
            last_loc, stay, begin, end, duration = tokens[0], float(tokens[1]), tokens[2], tokens[3], float(tokens[4])
            dd_b, hh_b, mm_b, loc_b = begin.split(":")
            dd_e, hh_e, mm_e, loc_e = end.split(":")

            # original gis
            if loc_b in converter.keys(): loc_b = converter[loc_b]
            if loc_e in converter.keys(): loc_e = converter[loc_e]
            if last_loc in converter.keys(): last_loc = converter[last_loc]
            if loc_b not in rtsmap.keys() or loc_e not in rtsmap.keys(): continue
            gis_begin, gis_end = rtsmap[loc_b], rtsmap[loc_e]

            if verbose: print tokens
            if duration > 0 and last_gis and last_hh and last_mm:
                if verbose: print "\t",str(loc_b)+"->"+str(loc_e),stay
                intervals.append((loc_b, loc_e, duration))

            last_dd, last_hh, last_mm, last_loc, last_gis = dd_e, hh_e, mm_e, loc_b, gis_end
            cnt+=1
        if intervals:
            distributions = group_by_segment(intervals)
            write_edges(distributions, train_edge_path+"/"+f)
    print "generate", cnt, "edge attribute data"


def gen_interval_data(rtsmap, converter, train_session_path, train_interval_path, thr_dist, thr_day, thr_trip):
    if not os.path.exists(train_interval_path): os.makedirs(train_interval_path+"/day"+thr_day)

    onlyfiles = [ f for f in listdir(train_session_path) if isfile(join(train_session_path,f)) ]
    #onlyfiles = ['6581804']
    pts, intervals = [], []
    cnt = 1
    for f in onlyfiles:
        sessionfile = open(train_session_path+"/"+f, "r")
        print "generating static intervals for", f
        last_dd, last_hh, last_mm, last_gis = "", "", "", ""
        pts, intervals, days = [], [], []
        for line in sessionfile:
            tokens = line.strip().split("\t")
            if len(tokens) < 4: print len(tokens), line.strip()
            last_loc, stay, begin, end, duration = tokens[0], float(tokens[1]), tokens[2], tokens[3], float(tokens[4])
            dd_b, hh_b, mm_b, loc_b = begin.split(":")
            dd_e, hh_e, mm_e, loc_e = end.split(":")


            # original gis
            if loc_b in converter.keys(): loc_b = converter[loc_b]
            if loc_e in converter.keys(): loc_e = converter[loc_e]
            if last_loc in converter.keys(): last_loc = converter[last_loc]
            if loc_b not in rtsmap.keys() or loc_e not in rtsmap.keys(): continue
            gis_begin, gis_end = rtsmap[loc_b], rtsmap[loc_e]

            if verbose: print tokens
            if stay > 0 and last_gis and last_hh and last_mm:
                # filter out weekends
                if not check_date(dd_b) or not check_date(dd_e): continue
                if last_loc == loc_b:
                    if verbose: print "\t",last_loc,str(last_hh)+":"+str(last_mm),str(hh_b)+":"+str(mm_b),stay
                    intervals.append((last_gis[1], last_gis[0], last_hh, last_mm, hh_b, mm_b))
                    if last_dd not in days: days.append(last_dd)
                else:
                    gis = []
                    gis.append([float(last_gis[1]), float(last_gis[0])])
                    gis.append([float(gis_begin[1]), float(gis_begin[0])])
                    tmp = numpy.array(gis)
                    if numpy.linalg.norm(tmp[0] - tmp[1]) < thr_dist:
                        intervals.append((last_gis[1], last_gis[0], last_hh, last_mm, hh_b, mm_b))
                        if last_dd not in days: days.append(last_dd)
                        if verbose: print "\t",str(last_loc)+"->"+str(loc_b),numpy.linalg.norm(tmp[0] - tmp[1]), gis[0], gis[1]

            last_dd, last_hh, last_mm, last_loc, last_gis = dd_e, hh_e, mm_e, loc_b, gis_end
            cnt+=1
        if intervals and len(intervals) > (thr_trip-1) and len(intervals) > (thr_day-1):
            write_intervals(intervals, train_interval_path+"/day"+thr_day+"/"+f)
    print "generate", cnt, "interval data"




def get_paths():
    paths = json.loads(open("/Users/mfchiang/Documents/lta/gmm/SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths


def main():
    usage = "%prog [options] <summary.csv>"
    version = "%prog 0.1"
    oparser = optparse.OptionParser(usage=usage, version=version)
    oparser.add_option('--train', dest='train', action='store_true', help = 'Train')
    oparser.add_option('--session', dest='session', action='store_true', help = 'Session')
    oparser.add_option('--numTripPlot', dest='numTripPlot', action='store_true', help = 'numTripPlot')
    oparser.add_option('--graph', dest='graph', action='store_true', help='Graph')
    oparser.add_option('--psudoss', dest='psudoss', action='store_true', help = 'Psudo Sessions')
    oparser.add_option('--action', dest='action', action='store_true', help = 'Actions')
    oparser.add_option('--interval', dest='interval', action='store_true', help = 'Intervals thr_dist')
    oparser.add_option('--nodeAttribute', dest='nodeAttribute', action='store_true', help = 'nodeAttribute')
    oparser.add_option('--edgeAttribute', dest='edgeAttribute', action='store_true', help = 'edgeAttribute')
    oparser.add_option('--test', dest='test', action='store_true', help = 'Test')
    oparser.add_option('--verbose', dest='verbose', action='store_true', default=False, help='Verbose')

    (options, args) = oparser.parse_args(sys.argv)
    if len(args) < 2: oparser.parse_args([sys.argv[0], "--help"])

    global verbose
    if options.verbose: verbose = True

    # 1.3839 103.8931 7 48
    if options.train:
        print("Preparing training data")
        traj_path = get_paths()["traj_path"]
        gis_path = get_paths()["gis_path"]
        gismap = read_gismap(gis_path)
        train_path = get_paths()["train_path"]
        test_path = get_paths()["test_path"]
        gen_train_data(gismap, traj_path, train_path, test_path, 200)

    # 1.3839 103.8931 7 48 locID, 1.2815 103.8409 8 55 locID
    if options.session:
        print("Preparing session data")
        traj_rts_path = get_paths()["traj_rts_path"]
        rts_gis_path = get_paths()["rts_gis_path"]
        rtsmap, converter = read_rts_gismap(rts_gis_path)
        train_session_path = get_paths()["train_session_path"]
        train_review_path = get_paths()["train_review_path"]
        test_session_path = get_paths()["test_session_path"]
        gen_session_data(rtsmap, converter, traj_rts_path, train_session_path, test_session_path, train_review_path, 15, 100)

    if options.numTripPlot:
        print("Preparing numTripPlot")
        traj_rts_path = get_paths()["traj_rts_path"]
        tripCnt_dist_path = get_paths()["tripCnt_dist_path"]
        data = count_session_data(traj_rts_path, tripCnt_dist_path, 15)
        plot_curve(data, "Num RTS Trips per User", "numTrip(RTS)", "numTripPlot")

    if options.graph:
        print("Preparing network data")
        rts_gis_path = get_paths()["rts_gis_path"]
        rtsmap, converter = read_rts_gismap(rts_gis_path)
        train_session_path = get_paths()["train_session_path"]
        train_graph_path = get_paths()["train_graph_path"]
        gen_graph_data(rtsmap, converter, train_session_path, train_graph_path)


    if options.nodeAttribute:
        thr_dist = float(sys.argv[2])
        print("Preparing nodeAttribute data")
        rts_gis_path = get_paths()["rts_gis_path"]
        rtsmap, converter = read_rts_gismap(rts_gis_path)
        train_session_path = get_paths()["train_session_path"]
        train_node_path = get_paths()["train_node_path"]
        gen_node_data(rtsmap, converter, train_session_path, train_node_path, thr_dist)


    if options.edgeAttribute:
        print("Preparing edgeAttribute data")
        rts_gis_path = get_paths()["rts_gis_path"]
        rtsmap, converter = read_rts_gismap(rts_gis_path)
        train_session_path = get_paths()["train_session_path"]
        train_edge_path = get_paths()["train_edge_path"]
        gen_edge_data(rtsmap, converter, train_session_path, train_edge_path)


    # lat lon hh mm
    if options.psudoss:
        sampling_ratio = [15,30,60,120]
        print("Preparing psudo session data")
        rts_gis_path = get_paths()["rts_gis_path"]
        rtsmap, converter = read_rts_gismap(rts_gis_path)
        train_session_path = get_paths()["train_session_path"]
        train_psudoss_path = get_paths()["train_psudoss_path"]
        train_path = get_paths()["train_path"]
        test_psudoss_path = get_paths()["test_psudoss_path"]
        print train_psudoss_path
        print test_psudoss_path
        gen_psudoss_data(rtsmap, converter, sampling_ratio, train_session_path, train_psudoss_path, test_psudoss_path, train_path)

    # boarding/alighting
    if options.action:
        print("Preparing action data")
        train_session_path = get_paths()["train_session_path"]
        train_action_path = get_paths()["train_action_path"]
        test_action_path = get_paths()["test_action_path"]
        gen_action_data(train_session_path, train_action_path, test_action_path)


    if options.interval:
        thr_dist = float(sys.argv[2])
        thr_day, thr_trip = 15, 30
        print("Preparing psudo session data")
        rts_gis_path = get_paths()["rts_gis_path"]
        rtsmap, converter = read_rts_gismap(rts_gis_path)
        train_session_path = get_paths()["train_session_path"]
        train_interval_path = get_paths()["train_interval_path"]
        print train_session_path
        print train_interval_path
        gen_interval_data(rtsmap, converter, train_session_path, train_interval_path, thr_dist, thr_day, thr_trip)


    if options.test:
        num = math.ceil(41.0/15.0)
        print int(num)
        for i in xrange(int(num)):
            delta = (i+1)*15
            if delta < 41: print (i+1)*15, time_addition(3,23,51,delta)
            else: print 41, time_addition(3,23,51,41).split(" ")

if __name__=="__main__":
    main()
