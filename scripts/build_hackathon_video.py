"""Render the three-minute Fleet Memory film from local recordings and event logs.

No policy or experiment code is imported or changed. The film describes the logged
17-dimensional v3.1 experiment. Illustrations are explicitly marked on screen.
Run: .venv/bin/python scripts/build_hackathon_video.py [--preview | --render]
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/fleet_memory_demo'
WORK = OUT / 'production'
WORK.mkdir(parents=True, exist_ok=True)
TIMELINE = json.loads((OUT / 'timeline.json').read_text())
W, H, FPS = 1920, 1080, 30
BG = '#0B111B'
PANEL = '#131E2C'
LINE = '#293A4C'
WHITE = '#F2F5F1'
MUTED = '#A7B8C9'
MINT = '#90E9C1'
BLUE = '#8CCBFF'
PURPLE = '#BEAFFA'
AMBER = '#F4CA84'
FONTS = '/Library/Fonts/Figtree-'


def read_log(p):
    result = []
    for line in (ROOT / p).read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


BENCH = read_log('logs/hopper/benchmark/events.jsonl')
SI = 'si_0__libero_spatial_0_plus_robot_init_0'
VERSIONS = {r['version']: r for r in BENCH if r.get('type') == 'skill_instance' and r.get('skill_instance_id') == SI}
CONS = [r for r in BENCH if r.get('type') == 'consolidation' and r.get('skill_instance_id') == SI and r['phase'] == 'end']
GATE = VERSIONS[3]['gate']
REPS = [r for r in read_log('logs/hopper/benchmark/reps.jsonl') if r.get('type') == 'benchmark_reps' and r['task'] == '0']
assert [(r['results']['BM-3']['k'], r['results']['BM-3']['n']) for r in REPS] == [(25,50),(29,50),(35,50)]
assert sum(r['results']['BM-1']['k'] for r in REPS) == 33
assert GATE['passed'] and GATE['n_seeds'] == 24
VIDEO_EVENTS = {r['episode_id']:r for r in read_log('logs/hopper/videos/events.jsonl') if r.get('type') == 'episode'}
PLAN = next(r['plan'] for r in read_log('logs/hopper/armC/events_v2.jsonl') if r.get('type') == 'episode')


@functools.lru_cache(None)
def font(size, style='Regular'):
    if style == 'Mono':
        return ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', size)
    return ImageFont.truetype(FONTS + style + '.ttf', size)


def ease(x):
    x = min(1., max(0., x))
    return x*x*(3-2*x)


def mix(a,b,t):
    return a+(b-a)*t


def txt(im, x, y, s, size=30, color=WHITE, style='Regular', anchor=None):
    d = ImageDraw.Draw(im)
    d.text((round(x),round(y)),str(s),font=font(size,style),fill=color,anchor=anchor)


def paragraph(im, x,y,s,width,size=30,color=MUTED,style='Regular',gap=1.28):
    words=s.split(); lines=[]; line=''
    for word in words:
        candidate=(line+' '+word).strip()
        if font(size,style).getlength(candidate)>width and line:
            lines.append(line);line=word
        else:line=candidate
    if line:lines.append(line)
    for i,line in enumerate(lines):txt(im,x,y+i*size*gap,line,size,color,style)
    return y+len(lines)*size*gap


def rect(im, box, fill=PANEL, outline=None, radius=18, width=2):
    ImageDraw.Draw(im).rounded_rectangle(tuple(round(v) for v in box),radius=radius,fill=fill,outline=outline,width=width)


def pill(im,x,y,label,color=MINT,size=23):
    w=font(size,'SemiBold').getlength(label)+34
    rect(im,(x,y,x+w,y+size+24),PANEL,color,12,1)
    txt(im,x+17,y+10,label,size,color,'SemiBold')
    return w


def arrow(im,x1,y1,x2,y2,color=MINT,width=3):
    d=ImageDraw.Draw(im);d.line((x1,y1,x2,y2),fill=color,width=width)
    angle=math.atan2(y2-y1,x2-x1)
    pts=[(x2,y2),(x2-14*math.cos(angle-.45),y2-14*math.sin(angle-.45)),(x2-14*math.cos(angle+.45),y2-14*math.sin(angle+.45))]
    d.polygon(pts,fill=color)


def pulse_path(im,points,t,color=MINT):
    d=ImageDraw.Draw(im)
    d.line(points,fill=LINE,width=4,joint='curve')
    lens=[math.dist(a,b) for a,b in zip(points,points[1:])]
    dist=(t%1)*sum(lens)
    for a,b,length in zip(points,points[1:],lens):
        if dist<=length:
            p=(mix(a[0],b[0],dist/length),mix(a[1],b[1],dist/length));
            d.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill=color);break
        dist-=length


def base(t,idx,sub):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    for j in range(3):rect(im,(64,35+j*10,86+j*8,40+j*10),MINT,radius=2)
    txt(im,112,31,'FLEET MEMORY',27,WHITE,'Bold')
    txt(im,392,36,'RECORDED EXPERIMENT  /  v3.1',19,MUTED,'Medium')
    txt(im,1856,34,f'{int(t)//60:02d}:{int(t)%60:02d} / 03:00',23,MUTED,'Mono',anchor='ra')
    title=TIMELINE[idx]['title']
    size=56
    while font(size,'SemiBold').getlength(title)>1790:size-=1
    txt(im,64,98,title,size,WHITE,'SemiBold')
    txt(im,66,177,sub,27,MUTED)
    y=1017
    for i,s in enumerate(TIMELINE):
        x=64+s['start']/180*1792;end=64+s['end']/180*1792-5
        d.line((x,y,end,y),fill=MINT if i<idx else LINE,width=4)
        if i==idx:d.line((x,y,mix(x,end,(t-s['start'])/(s['end']-s['start'])),y),fill=MINT,width=4)
    txt(im,64,1034,f'{idx+1:02d}  {TIMELINE[idx]["chapter"]}',18,MINT,'Bold')
    txt(im,1856,1034,'SmolVLA  ·  LIBERO / LIBERO-Plus  ·  GMU Hopper',18,MUTED,anchor='ra')
    return im


def takeaway(im,s,color=MINT):
    rect(im,(64,933,1856,990),'#182638',None,13)
    size=31
    while font(size,'Medium').getlength(s)>1730:size-=1
    txt(im,960,951,s,size,color,'Medium',anchor='ma')


CLIP_DATA={}
CLIP_ACTION={'BM-0_ok':3.5,'BM-1_fail':11.05,'BM-3_ok':4.75,'MASTERY-B_ok':10.25}


def load_clips():
    for name in CLIP_ACTION:
        path=ROOT / 'logs/hopper/videos' / (name+'.mp4')
        b=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'])
        a=np.frombuffer(b,np.uint8).reshape(-1,256,512,3)
        CLIP_DATA[name]=[Image.fromarray(frame) for frame in a]
        print('Loaded',name,len(a),'frames',flush=True)


@functools.lru_cache(72)
def frame_resized(name,n,cam,w,h):
    # Remove the recorder's tiny text; all new labels derive from the event log.
    box={'both':(0,34,512,234),'external':(0,34,256,234),'wrist':(256,34,512,234)}[cam]
    pic=CLIP_DATA[name][n].crop(box)
    ratio=min(w/pic.width,h/pic.height)
    return pic.resize((round(pic.width*ratio),round(pic.height*ratio)),Image.Resampling.BICUBIC)


def footage(im,name,sec,box,cam='external',label=None,accent=MINT):
    x,y,w,h=map(round,box)
    n=min(len(CLIP_DATA[name])-1,max(0,int(min(sec,CLIP_ACTION[name]-.051)*20)))
    rect(im,(x-2,y-2,x+w+2,y+h+2),PANEL,LINE,12)
    pic=frame_resized(name,n,cam,w,h)
    im.paste(pic,(x+(w-pic.width)//2,y+(h-pic.height)//2))
    if label:
        rect(im,(x+14,y+14,x+min(w-14,font(21,'SemiBold').getlength(label)+45),y+57),BG,None,9)
        txt(im,x+28,y+25,label,21,accent,'SemiBold')


def source(im,s,x=64,y=901):txt(im,x,y,s,19,MUTED)


def opening(t,u,idx):
    im=base(t,idx,'Put the black bowl between the plate and ramekin onto the plate.')
    sec=u*.43 if idx==0 else u*.84
    footage(im,'BM-3_ok',sec,(64,236,884,646),'external','EXTERNAL CAMERA  ·  RECORDED REPLAY')
    footage(im,'BM-3_ok',sec,(972,236,884,646),'wrist','WRIST CAMERA  ·  RECORDED REPLAY')
    source(im,'LIBERO-Spatial task 0  ·  seed 5048  ·  approved v2 settings')
    if sec>=4.4:
        pill(im,706,795,'SIMULATOR CONFIRMED SUCCESS',MINT,26)
    takeaway(im,'A fixed robot model, guided by a saved memory file.' if idx==0 else 'Fixed model  +  tested memory  →  next attempt')
    return im


def policy(t,u,idx):
    im=base(t,idx,'A “policy” is the controller that chooses the robot’s next movement.')
    footage(im,'BM-3_ok',(u*.55)%4.7,(64,260,530,416),'external','CAMERA INPUT')
    footage(im,'BM-3_ok',(u*.55)%4.7,(64,701,530,181),'wrist',None)
    rect(im,(657,283,1153,636),PANEL,PURPLE)
    txt(im,905,311,'SmolVLA',55,WHITE,'SemiBold',anchor='ma')
    txt(im,905,380,'vision → language → action',27,PURPLE,anchor='ma')
    pill(im,766,438,'WEIGHTS FIXED',PURPLE,25)
    paragraph(im,701,520,'Already learned how to turn observations into actions.',410,30)
    pulse_path(im,[(598,472),(653,472)],u*.7,BLUE)
    pulse_path(im,[(1158,472),(1230,472)],u*.7,MINT)
    txt(im,660,695,'INSTRUCTION',21,BLUE,'Bold')
    paragraph(im,660,736,'“Pick up the black bowl and place it on the plate.”',500,30,WHITE)
    txt(im,660,835,'+ current hand position and orientation',24,MUTED)
    txt(im,1245,260,'ONE ACTION = 7 NUMBERS',27,MINT,'Bold')
    labels=['move x','move y','move z','rotate x','rotate y','rotate z','gripper']
    for j,label in enumerate(labels):
        yy=319+j*74; col=BLUE if j<3 else PURPLE if j<6 else MINT
        txt(im,1245,yy,label,27,col)
        rect(im,(1410,yy+3,1835,yy+34),PANEL,None,6)
        center=1622; val=math.sin(u*1.5+j*1.2)*.8 if j<6 else (1 if (u%5)>2 else -1)
        ImageDraw.Draw(im).line((center,yy,center,yy+38),fill=LINE,width=2)
        rect(im,(min(center,center+val*193),yy+7,max(center+1,center+val*193),yy+29),col,None,4)
    source(im,'Command display is a schematic illustration.',1245,864)
    takeaway(im,'The model predicts short sequences of 10 movement commands.')
    return im


def compare(t,u,idx):
    im=base(t,idx,'Same task, same changed start, same seed 5048, same frozen SmolVLA.')
    sec=min(11.0,u*.90)
    footage(im,'BM-1_fail',sec,(64,276,868,603),'external','ORIGINAL CONTROLLER',BLUE)
    footage(im,'BM-3_ok',sec,(988,276,868,603),'external','AFTER ONE SLEEP CYCLE',MINT)
    txt(im,65,231,'Before loading learned settings',27,BLUE,'SemiBold')
    txt(im,989,231,'Load approved settings v2',27,MINT,'SemiBold')
    if u>5.4:pill(im,1100,794,'TASK COMPLETE  ·  94 CONTROL STEPS',MINT,24)
    if u>12.3:pill(im,195,794,'ORIGINAL RUN ENDS AT 220 STEPS',BLUE,23)
    source(im,'Joint offset range r = 0.1 radians ≈ 5.7°  ·  actual recorded comparison; completed view holds')
    takeaway(im,'The learned internal model stays fixed. The saved execution settings change.')
    return im


def trace(t,u,idx):
    im=base(t,idx,'Follow the path every movement command takes.')
    footage(im,'BM-3_ok',(u*.4)%4.7,(65,245,835,652),'external','ROBOT EXECUTION  ·  REPLAY')
    nodes=[(1030,253,1800,378,'1   FROZEN POLICY','Predict the next movements',PURPLE),
           (1030,468,1800,611,'2   EXECUTION LAYER','Apply the approved settings file',MINT),
           (1030,709,1800,852,'3   SAFETY ENVELOPE','Clamp commands + workspace bounds',BLUE)]
    for x,y,x2,y2,title,desc,col in nodes:
        rect(im,(x,y,x2,y2),PANEL,col)
        txt(im,x+27,y+25,title,29,col,'Bold');txt(im,x+27,y+76,desc,28,WHITE)
    pulse_path(im,[(1415,381),(1415,462)],u*.6)
    pulse_path(im,[(1415,614),(1415,705)],u*.6)
    pulse_path(im,[(1026,786),(951,786),(951,588),(904,588)],u*.6,BLUE)
    pill(im,1510,408,'VERSIONED MEMORY',MINT,19)
    takeaway(im,'Safety limits apply after the execution layer, on every command.')
    return im


def settings(t,u,idx):
    im=base(t,idx,'17 bounded values per task and environment. “Bounded” means each has a fixed allowed range.')
    footage(im,'BM-3_ok',min(u*.28,4.7),(64,247,786,615),'external','HOMING → POLICY CONTROL  ·  v2 REPLAY')
    txt(im,66,884,'Homing restores a reference hand position and orientation.',24,MUTED)
    groups=[('APPROACH',6,BLUE),('TIMING / LIMITS / BLEND',4,PURPLE),('HOMING',7,MINT)]
    xx=925
    for title,n,col in groups:
        for j in range(n):
            h=45+20*math.sin(u*.7+j)
            rect(im,(xx,347-h,xx+38,362),col,None,5);xx+=48
    txt(im,923,237,'6 approach + 4 execution + 7 homing = 17',30,WHITE,'SemiBold')
    p=VERSIONS[3]['params']; v=VERSIONS[2]['params']
    a=ease((u-2)/9)
    rows=[('Action timing',v['time_scale'],p['time_scale'],.6,1.5,PURPLE),
          ('Speed cap',v['velocity_cap'],p['velocity_cap'],.3,1.,BLUE),
          ('Gripper close command',v['gripper_cmd'],p['gripper_cmd'],.3,1.,MINT)]
    for j,(name,v0,v1,lo,hi,col) in enumerate(rows):
        y=397+j*135;val=mix(v0,v1,a)
        txt(im,927,y,name,28,WHITE,'Medium');txt(im,1817,y,f'{val:.3f}',30,col,'Mono',anchor='ra')
        rect(im,(930,y+56,1809,y+66),LINE,None,4)
        x=930+(val-lo)/(hi-lo)*879
        ImageDraw.Draw(im).ellipse((x-10,y+51,x+10,y+71),fill=col)
        txt(im,930,y+78,f'{lo:.2f}',17,MUTED,'Mono');txt(im,1809,y+78,f'{hi:.2f}',17,MUTED,'Mono',anchor='ra')
    pill(im,928,831,'HOMING: ON',MINT,25)
    txt(im,1193,843,'Saved values: v2 → v3',25,MUTED)
    takeaway(im,'Timing below 1 stretches the action sequence. Homing happens before the policy acts.')
    return im


def coach(t,u,idx):
    im=base(t,idx,'S1 stores structured plans and lessons. S3 stores numerical execution settings.')
    rect(im,(64,247,923,884),PANEL,PURPLE)
    rect(im,(959,247,1856,884),PANEL,MINT)
    txt(im,94,275,'S1  /  LANGUAGE-MODEL COACH',29,PURPLE,'Bold')
    txt(im,990,275,'S3  /  NUMERICAL OPTIMIZER',29,MINT,'Bold')
    txt(im,95,333,'Recorded plan: bowl into drawer, then close',25,MUTED)
    steps=[s['skill'].upper() for s in PLAN['subtasks']]
    n=min(4,int(u/2.6))
    for j,name in enumerate(steps):
        yy=399+j*65
        rect(im,(95,yy,880,yy+48),'#202E43' if j==n else BG,PURPLE if j==n else None,8)
        txt(im,115,yy+8,f'{j+1:02d}    {name}',25,PURPLE if j==n else WHITE,'Medium')
    txt(im,95,755,'candidate → A/B test → validated',27,PURPLE,'SemiBold')
    txt(im,95,808,'≥30 matched trials  ·  p<0.05  ·  lift ≥2 points',23,MUTED)
    # This is a memory mechanism illustration, not footage of a coach intervention.
    footage(im,'MASTERY-B_ok',min(u*.7,10.2),(990,342,831,324),'both','SEPARATE EXECUTION REPLAY')
    paragraph(im,992,700,'Search proposes settings. A separate gate approves a new version.',800,31,WHITE)
    txt(im,992,814,'Success source: Env.success_flag()',25,MINT,'Mono')
    takeaway(im,'Lessons are selected before reset. Plans are structured edits the program can apply.')
    return im


def monitor(t,u,idx):
    im=base(t,idx,'Preserve every attempt; watch recent performance for a meaningful change.')
    rect(im,(64,247,886,889),PANEL,LINE)
    txt(im,96,278,'EVENT LOG',27,MINT,'Bold')
    txt(im,96,324,'One new record at a time. History stays intact.',25,MUTED)
    ep=VIDEO_EVENTS['ep_2131e0e0']
    lines=[('episode_id',ep['episode_id']),('seed',str(ep['seed'])),('policy','smolvla-450m'),('settings','approved v2'),('steps',str(ep['outcome']['steps'])),('env_success','true')]
    visible=min(6,1+int(u/1.1))
    for j,(k,v) in enumerate(lines[:visible]):
        yy=397+j*61
        txt(im,98,yy,k,27,BLUE,'Mono');txt(im,454,yy,v,27,MINT if k=='env_success' else WHITE,'Mono')
    txt(im,97,816,'Replay of recorded episode ep_2131e0e0',22,MUTED)
    txt(im,954,272,'DRIFT DETECTOR',27,BLUE,'Bold')
    txt(im,954,325,'Recent attempts count more.',32,WHITE,'Medium')
    # Explicitly schematic: avoid implying the headline benchmark was drift-triggered.
    d=ImageDraw.Draw(im);pts=[]
    vals=[.28+.02*math.sin(j*1.9)+max(0,j-13)*.019 for j in range(30)]
    ew=[];v=vals[0]
    for value in vals:v=.3*value+.7*v;ew.append(v)
    n=min(30,max(2,int(u*2.7)))
    for j in range(n):pts.append((981+j*28,720-ew[j]*480))
    d.line((979,425,979,754,1828,754),fill=LINE,width=2)
    d.line((980,505,1828,505),fill='#607181',width=2)
    txt(im,1809,468,'trigger level',22,MUTED,anchor='ra')
    d.line(pts,fill=BLUE,width=5,joint='curve')
    p=pts[-1];d.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill=BLUE)
    txt(im,981,767,'attempts →',21,MUTED)
    pill(im,1030,824,'AUTO-START SLEEP' if n>25 else 'MONITOR RECENT RESULTS',MINT,25)
    source(im,'Drift chart: mechanism illustration.',954,892)
    takeaway(im,'EWMA = a moving average weighted toward the most recent attempts.')
    return im


def search(t,u,idx):
    im=base(t,idx,'Cross-entropy search: try settings → keep the best → narrow the next search.')
    rect(im,(64,244,1154,875),PANEL,LINE)
    iteration=min(3,int(u/4.2));phase=(u%4.2)/4.2
    txt(im,94,273,f'BATCH {iteration+1} OF 4',28,MINT,'Bold')
    txt(im,1123,278,'24 candidates  ·  keep 6',25,MUTED,anchor='ra')
    # Conceptual grid, with actual batch sizes from cons_02b6cf57.
    rng=np.random.default_rng(110+iteration)
    values=rng.random((24,17))
    elites={1,5,8,14,17,21}
    for j in range(24):
        col=j%6;row=j//6;x=96+col*174;y=354+row*108
        selected=j in elites and phase>.45
        rect(im,(x,y,x+152,y+87),'#223C38' if selected else BG,MINT if selected else LINE,8)
        txt(im,x+11,y+7,f'{j+1:02d}',18,MINT if selected else MUTED,'Mono')
        for k,v in enumerate(values[j]):
            xx=x+11+k*7.6;hh=10+v*31
            ImageDraw.Draw(im).line((xx,y+72,xx,y+72-hh),fill=MINT if selected else '#5E7F9F',width=4)
    txt(im,96,814,'Illustration of candidate selection; each tiny bar is one setting.',21,MUTED)
    txt(im,1210,257,'HOW AN ATTEMPT IS SCORED',26,BLUE,'Bold')
    terms=[('1.0 × time','steps relative to a fixed baseline',BLUE),('0.5 × abrupt motion','how sharply movement changes',PURPLE),('0.5 × command size proxy','large movement commands',AMBER),('3.0 × incomplete task','from the simulator’s success flag',MINT)]
    for j,(a,b,col) in enumerate(terms):
        yy=334+j*111
        txt(im,1210,yy,a,31,col,'SemiBold');txt(im,1210,yy+46,b,23,MUTED)
    pill(im,1213,801,'LOWER COST IS BETTER',MINT,25)
    if u>14:
        pill(im,277,759,'FINALISTS → FRESH TRIALS',MINT,25)
    source(im,'Recorded cycle 2: 24 candidates × 4 rounds; 464 total simulator rollouts including validation + gate.')
    takeaway(im,'The search updates the 17-number file. The model’s weights stay fixed.')
    return im


def gate(t,u,idx):
    im=base(t,idx,'An evaluation layout is a starting scene. The three groups never overlap.')
    specs=[(64,628,'SEARCH','layouts 00–29',0,30,BLUE),(658,1222,'GATE','layouts 30–39',30,10,PURPLE),(1252,1856,'FINAL EVALUATION','layouts 40–49',40,10,MINT)]
    for x,x2,name,span,start,n,col in specs:
        rect(im,(x,245,x2,558),PANEL,col)
        txt(im,x+25,271,name,26,col,'Bold');txt(im,x+25,317,span,25,WHITE)
        step=(x2-x-50)/10
        for j in range(n):
            xx=x+25+(j%10)*step;yy=371+(j//10)*49
            lit=(j+start)<u*5
            rect(im,(xx,yy,xx+step-7,yy+38),col if lit else BG,None,5)
            txt(im,xx+(step-7)/2,yy+8,f'{start+j:02}',18,BG if lit else MUTED,'Mono',anchor='ma')
    rect(im,(64,600,941,878),PANEL,MINT)
    rect(im,(977,600,1856,878),PANEL,MINT)
    txt(im,99,628,'REQUIREMENT 1',23,MINT,'Bold')
    txt(im,99,679,'Cost must go down',43,WHITE,'SemiBold')
    txt(im,99,756,'candidate cost < current cost',29,BLUE,'Mono')
    txt(im,1012,628,'REQUIREMENT 2',23,MINT,'Bold')
    txt(im,1012,679,'Preserve success',43,WHITE,'SemiBold')
    txt(im,1012,756,'candidate ≥ current − 2 points',29,BLUE,'Mono')
    txt(im,1012,817,'“Points” means percentage points.',23,MUTED)
    source(im,'The gate compares both files on matched trials. Final evaluation is kept separate.')
    takeaway(im,'Search never sees the gate layouts. Neither search nor gate sees the evaluation layouts.')
    return im


def ledger(t,u,idx):
    im=base(t,idx,'Replay of the recorded second-cycle promotion for the bowl-placement environment.')
    rect(im,(64,246,1044,888),PANEL,LINE)
    txt(im,99,277,'skill_instance  /  saved record',29,MINT,'Mono')
    fields=[('version','3'),('parent_version','2'),('status','incumbent  =  active'),('produced_by','sleep optimizer'),('gate.passed','true'),('gate.n_seeds','24'),('time_scale',f'{VERSIONS[3]["params"]["time_scale"]:.3f}')]
    visible=min(7,1+int(u/1.2))
    for j,(k,v) in enumerate(fields[:visible]):
        yy=363+j*65;txt(im,103,yy,k,29,BLUE,'Mono');txt(im,536,yy,v,28,MINT if j in [2,4] else WHITE,'Mono')
    txt(im,1109,254,'GATE: BOTH CHECKS PASSED',27,MINT,'Bold')
    txt(im,1109,319,'Mean cost',27,MUTED)
    txt(im,1109,367,f'{GATE["incumbent_cost"]:.3f}  →  {GATE["candidate_cost"]:.3f}',53,WHITE,'SemiBold')
    txt(im,1109,466,'Success on 24 gate trials',27,MUTED)
    txt(im,1109,514,f'{100*GATE["incumbent_success"]:.1f}%  →  {100*GATE["candidate_success"]:.1f}%',53,MINT,'SemiBold')
    for j,label in enumerate(['v1','v2','v3']):
        xx=1163+j*247
        rect(im,(xx,698,xx+139,810),PANEL,MINT if j==2 else LINE,16)
        txt(im,xx+69,718,label,46,MINT if j==2 else MUTED,'SemiBold',anchor='ma')
        if j<2:arrow(im,xx+145,752,xx+236,752,MINT)
    pill(im,1628,831,'ACTIVE',MINT,22)
    source(im,'Source: cons_02b6cf57  ·  skill_instance v3  ·  benchmark/events.jsonl')
    takeaway(im,'Append the new version and its evidence; keep one active version per task + environment.')
    return im


def results(t,u,idx):
    im=base(t,idx,'LIBERO-Spatial task 0  ·  LIBERO-Plus robot-start perturbation  ·  frozen SmolVLA')
    rect(im,(64,246,1225,891),PANEL,LINE)
    txt(im,101,276,'SUCCESS ON UNSEEN EVALUATION LAYOUTS',26,MINT,'Bold')
    values=[22,54,70];ci=[(16,29),(44,64),(56,81)]
    labels=['Before sleep','One cycle · v2','Two cycles · v3']
    ns=['33 / 150 trials','54 / 100 trials','35 / 50 trials']
    cols=[BLUE,PURPLE,MINT];bottom=730
    for j in range(3):
        xx=173+j*347;p=ease((u-j*2)/2.2);height=values[j]*4.7*p
        rect(im,(xx,bottom-height,xx+197,bottom),cols[j],None,9)
        if p>.96:
            ylo=bottom-ci[j][0]*4.7;yhi=bottom-ci[j][1]*4.7;cx=xx+98
            ImageDraw.Draw(im).line((cx,ylo,cx,yhi),fill=WHITE,width=3)
            ImageDraw.Draw(im).line((cx-13,ylo,cx+13,ylo),fill=WHITE,width=3)
            ImageDraw.Draw(im).line((cx-13,yhi,cx+13,yhi),fill=WHITE,width=3)
        txt(im,xx+98,bottom-height-88,f'{round(values[j]*p)}%',65,cols[j],'SemiBold',anchor='ma')
        txt(im,xx+98,755,labels[j],27,WHITE,'Medium',anchor='ma')
        txt(im,xx+98,799,ns[j],25,MUTED,anchor='ma')
    txt(im,101,852,'10 unseen layouts × repeated trials  ·  whiskers: 95% confidence intervals',22,MUTED)
    footage(im,'BM-3_ok',(u*.55)%4.7,(1270,247,586,460),'external','ONE-CYCLE REPLAY  ·  v2')
    txt(im,1277,744,'Same model.',38,WHITE,'SemiBold')
    txt(im,1277,796,'New approved file.',38,MINT,'SemiBold')
    source(im,'Results: benchmark/reps.jsonl + docs/RESULTS.md  ·  Replay: seed 5048; aggregate bars use repeated evaluations.')
    takeaway(im,'22% → 54% → 70% success across two unattended sleep cycles.')
    return im


DRAWERS=[opening,policy,compare,trace,settings,coach,monitor,search,gate,ledger,results,opening]


def draw_frame(t):
    idx=next((i for i,s in enumerate(TIMELINE) if s['start']<=t<s['end']),11)
    u=t-TIMELINE[idx]['start']
    im=DRAWERS[idx](t,u,idx)
    # Short editorial fade at scene boundaries, maintaining the full 180-second timeline.
    if idx>0 and u<.20:
        overlay=Image.new('RGB',(W,H),BG)
        im=Image.blend(overlay,im,.40+.60*ease(u/.20))
    if t>179.5:
        im=Image.blend(im,Image.new('RGB',(W,H),BG),ease((t-179.5)/.5)*.88)
    return im


def stamp(sec,ms=False):
    s=int(sec)
    if ms:return f'{s//3600:02d}:{s//60%60:02d}:{s%60:02d},{round((sec-s)*1000):03d}'
    return f'{s//60:02d}:{s%60:02d}'


def artifacts():
    md=['# Fleet Memory — three-minute narration','',
        'Narration-ready picture: **Fleet_Memory_3min.mp4**. Exactly 3:00, 1920 × 1080, 30 fps. The master is silent so you can record your own voice.',
        '', 'Read conversationally at about 143 words per minute. Each paragraph starts at the timestamp shown; use the remaining space for a brief pause. Pronounce SmolVLA “small vee ell ay,” LIBERO “lee-BEAR-oh,” and S1/S3 “ess one / ess three.”', '']
    captions=[];counter=1;plain=[]
    for s in TIMELINE:
        md.extend([f'## {stamp(s["start"])}–{stamp(s["end"])} · {s["chapter"].title()}', '', s['narration'],''])
        plain.append(f'[{stamp(s["start"])} – {stamp(s["end"])}]\n{s["narration"]}\n')
        # Optional reference captions use the planned narration, not a recorded voice.
        words=s['narration'].split();groups=[];group=[]
        for word in words:
            group.append(word)
            if len(group)>=10 and (word.endswith(('.',':',';','?')) or len(group)>=15):groups.append(group);group=[]
        if group:
            if len(group)<5 and groups:groups[-1]+=group
            else:groups.append(group)
        cur=s['start']+.15;dur=s['end']-s['start']-.4
        for group in groups:
            end=cur+dur*len(group)/len(words)
            captions.append(f'{counter}\n{stamp(cur,True)} --> {stamp(end,True)}\n'+textwrap.fill(' '.join(group),62)+'\n')
            counter+=1;cur=end
    (OUT/'Narration.md').write_text('\n'.join(md))
    (OUT/'Narration.txt').write_text('\n'.join(plain))
    (OUT/'Narration.srt').write_text('\n'.join(captions))
    evidence={
        'scope':'Recorded Fleet Memory v3.1 17-dimensional robot-init experiment; current uncommitted code is not changed.',
        'main_video':'180 seconds; 1920x1080; 30 fps; no audio; source footage enlarged and cropped to remove tiny recorder overlays.',
        'clips':{},'headline':{'scope':'libero_spatial task 0, robot_init r=0.1, eval layouts 40-49','before':[33,150],'one_cycle':[54,100],'two_cycles':[35,50]},
        'gate':GATE,'version_3':VERSIONS[3],
        'illustrations':['Animated command values are schematic, not measured telemetry.','Search grid illustrates CEM; grid values are illustrative, batch sizes and total rollouts are recorded.','Drift graph is a mechanism illustration; the headline benchmark cycles were manually launched and then unattended.','The S1 plan is a real recorded planner output; adjacent replay is a separate optimizer-only drawer episode.'],
        'clip_reconciliation':'The current local MP4s were overwritten by the second recording run. Main bowl clips correspond to seed 5048, v2 success 94 steps; drawer clip seed 2, success 204 steps. RESULTS.md section 7 describes the earlier seed 5047/0 clips.',
        'sources':['docs/RESULTS.md','logs/hopper/videos/events.jsonl','logs/hopper/benchmark/events.jsonl','logs/hopper/benchmark/reps.jsonl','logs/hopper/armC/events_v2.jsonl','fleet_memory/execution/params.py','fleet_memory/runner/consolidate.py','fleet_memory/memory/lifecycle.py','fleet_memory/memory/drift.py','fleet_memory/analysis/cost.py']}
    mapping={'BM-0_ok':'ep_796838a6','BM-1_fail':'ep_597da1cd','BM-3_ok':'ep_2131e0e0','MASTERY-B_ok':'ep_18d30309'}
    for name,epid in mapping.items():
        path=ROOT/'logs/hopper/videos'/f'{name}.mp4'
        evidence['clips'][name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'episode_id':epid,'seed':VIDEO_EVENTS[epid]['seed'],'outcome':VIDEO_EVENTS[epid]['outcome']}
    (WORK/'sources.json').write_text(json.dumps(evidence,indent=2))
    chapters=[';FFMETADATA1','title=Fleet Memory — Three-minute technical demo',
              'comment=Silent narration-ready master. Recorded simulation plus explanatory animations.']
    for s in TIMELINE:
        chapters.extend(['[CHAPTER]','TIMEBASE=1/1000',f'START={s["start"]*1000}',f'END={s["end"]*1000}',
                         f'title={s["chapter"].title()}: {s["title"]}'])
    (WORK/'chapters.ffmeta').write_text('\n'.join(chapters)+'\n')


def preview():
    sheet=Image.new('RGB',(1440,4*312),BG)
    for i,s in enumerate(TIMELINE):
        t=s['start']+(s['end']-s['start'])*.64
        im=draw_frame(t);im.save(WORK/f'scene_{i+1:02d}.png')
        sm=im.resize((480,270),Image.Resampling.LANCZOS)
        x=(i%3)*480;y=(i//3)*312
        sheet.paste(sm,(x,y));txt(sheet,x+12,y+276,f'{stamp(s["start"])}  {s["chapter"]}',21,MUTED)
    sheet.save(OUT/'Storyboard.jpg',quality=94)
    draw_frame(164).save(OUT/'Poster.jpg',quality=95)
    print('Wrote preview + narration artifacts',flush=True)


def render():
    target=OUT/'Fleet_Memory_3min.mp4'
    cmd=['ffmpeg','-v','warning','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-',
         '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-threads','6',
         '-movflags','+faststart','-metadata','title=Fleet Memory — Three-minute technical demo',
         '-metadata','comment=Recorded LIBERO simulation with explanatory animations. Narration-ready silent master.',str(target)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    started=time.time()
    try:
        for i in range(180*FPS):
            proc.stdin.write(draw_frame(i/FPS).tobytes())
            if i%(FPS*10)==0:print(f'Rendered {i//FPS:03d}/180 seconds; elapsed {time.time()-started:.1f}s',flush=True)
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
    except BaseException:
        proc.kill();raise
    chaptered=WORK/'chaptered.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(target),'-i',str(WORK/'chapters.ffmeta'),
                    '-map_metadata','1','-map_chapters','1','-map','0:v:0','-c','copy',
                    '-movflags','+faststart',str(chaptered)],check=True)
    chaptered.replace(target)
    print('COMPLETE',target,target.stat().st_size,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview',action='store_true');p.add_argument('--render',action='store_true');args=p.parse_args()
    artifacts();load_clips()
    if args.preview or not args.render:preview()
    if args.render:render()
