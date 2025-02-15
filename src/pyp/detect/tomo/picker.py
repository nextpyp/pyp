from pyp.system.local_run import run_shell_command
from pyp.inout.image import mrc
from pyp.utils import timer
from pyp.system.utils import get_imod_path
from pyp.utils import get_relative_path
from pyp.system.logging import initialize_pyp_logger

import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage.morphology import remove_small_objects,ball

relative_path = str(get_relative_path(__file__))
logger = initialize_pyp_logger(log_name=relative_path)

def extract(image,boxes,boxsize):
    particles = np.zeros((len(boxes), boxsize, boxsize, boxsize))
    i=0
    for box in boxes:
        x,y,z= box[2],box[1],box[0]
        particles[i, :, :, :]=image[int(x-boxsize/2):int(x+boxsize/2),int(y-boxsize/2):int(y+boxsize/2),int(z-boxsize/2):int(z+boxsize/2)]
        i=i+1
    return particles

class Picker():

    @timer.Timer(
        "init", text="\tTotal time elapsed (picker init): {}", logger=logger.info
    )
    def __init__(self,name,radius=100,pixelsize=2.1,auto_binning = 12,show=False):
        self.radius=radius
        self.pixelsize=pixelsize
        self.name=name
        self.auto_binning = auto_binning

        self.rec = mrc.read(name+'.rec')
        self.tilesize = int(3 * radius / pixelsize/ auto_binning)
        if self.tilesize % 2 > 0:
            self.tilesize += 1

        radius1 = 0.001
        radius2 = 0.5 * self.pixelsize * self.auto_binning / self.radius
        sigma1 = 0
        sigma2 = 0.001

        com = "{5}/bin/mtffilter -3dfilter -radius1 {1} -hi {2} -l {3},{4} {0}.rec bp.mrc".format(
            self.name, radius1, sigma1, radius2, sigma2,get_imod_path()
        )


        run_shell_command(com)

        self.lowres = mrc.read("bp.mrc")

        if show:
            run_shell_command(get_imod_path()+"/bin/imod -xyz -Y "+name+".rec")
            run_shell_command(get_imod_path()+"/bin/imod -xyz -Y bp.mrc")

    @timer.Timer(
        "getcont", text="\tTotal time elapsed (picker getcont): {}", logger=logger.info
    )
    def getcont(self,contract_times=1,gaussian=True,sigma=15,stdtimes=3.5,min_size=125,dilation=100,show=False):
        G = self.rec.reshape(int(self.rec.shape[0]/contract_times),contract_times,int(self.rec.shape[1]/contract_times),contract_times,int(self.rec.shape[2]/contract_times),contract_times).mean(1).mean(2).mean(3)
        if gaussian:
            Gf = scipy.ndimage.gaussian_filter(G, sigma=sigma)
            G = G-Gf
            if show:
                mrc.write(scipy.ndimage.gaussian_filter(G, sigma=sigma), "gaussian.mrc")
                run_shell_command(get_imod_path()+"/bin/imod -xyz -Y gaussian.mrc")

        maskthres=G.mean()-stdtimes*G.std()
        mask = G < maskthres
        cmask = scipy.ndimage.morphology.binary_opening(mask, ball(1))
        clean = remove_small_objects(cmask, min_size=min_size)
        segmentation = scipy.ndimage.morphology.binary_closing(clean, ball(2))
        contamination_dilation = int(dilation/contract_times / self.pixelsize/ self.auto_binning)
        area = scipy.ndimage.morphology.binary_dilation(segmentation, ball(contamination_dilation))
        if show:
            mrc.write(area, "area.mrc")
            run_shell_command(get_imod_path()+"/bin/imod -xyz -Y area.mrc")

        return area

    @timer.Timer(
        "detect", text="\tTotal time elapsed (picker detect): {}", logger=logger.info
    )
    def detect(self,area,contract_times=1,radius_times=4,inhibit=False,detection_width=128,show=False):
        points=self.minima_extract(radius_times=radius_times,inhibit=inhibit)
        boxes = []
        for i in range(len(points)):
            x = points[i][0]
            y = points[i][1]
            z = points[i][2]

            clean = not area[int(x/contract_times),int(y/contract_times),int(z/contract_times)]
            inside = (
                x- self.tilesize / 2 >= 0
                and x < self.rec.shape[0] - self.tilesize/2 + 1
                and y- self.tilesize / 2 >= 0
                and y < self.rec.shape[1] - self.tilesize/2 + 1
                and z- self.tilesize / 2 >= 0
                and z < self.rec.shape[2] - self.tilesize/2 + 1
                and y <= self.rec.shape[1]*0.5+detection_width
                and y >= self.rec.shape[1]*0.5-detection_width
            )
            if clean and inside:
                boxes.append([z,y,x])

        logger.info(f"\t{len(boxes):,} initial positions found")

        if show:
            f = open('boxes.txt', 'w')
            f.writelines([' '.join(str(boxes[i])[1:-1].split(', '))+'\n' for i in range(len(boxes))])
            f.close()
            run_shell_command(get_imod_path()+'/bin/point2model boxes.txt boxes.mod -sp 5')
            run_shell_command(get_imod_path()+"/bin/imod -xyz -Y bp.mrc boxes.mod")


        raw_particles = extract(self.lowres, boxes,self.tilesize)
        return boxes,raw_particles

    @timer.Timer(
        "prefilt", text="\tTotal time elapsed (picker prefilt): {}", logger=logger.info
    )
    def prefilt(self,raw_particles,stdtimes=1):
        x, y= np.mgrid[0:self.tilesize, 0:self.tilesize] - self.tilesize / 2 + 0.5
        condition2d = np.sqrt(x*x+y*y) > self.radius / self.pixelsize / self.auto_binning
        raw_particles2d=raw_particles.sum(axis=2)

        particles_metrics = np.zeros([raw_particles.shape[0],2])
        for p in range(raw_particles.shape[0]):
            raw = raw_particles2d[p, :, :]
            background = np.extract(condition2d, raw)
            foreground = np.extract(np.logical_not(condition2d), raw)
            particles_metrics[p,0] =foreground.std()
            particles_metrics[p,1] =background.std()
        stdmean,stdstd=particles_metrics[:,0].mean(),particles_metrics[:,0].std()
        logger.info(f"\tPre-filtering image statistics: mean = {stdmean:.2f}, std = {stdstd:.2f}")

        stdthreshold=stdmean+stdstd*stdtimes
        return particles_metrics,stdthreshold

    @timer.Timer(
        "filt", text="\tTotal time elapsed (picker filt): {}", logger=logger.info
    )
    def filt(self,boxes,particles_metrics,stdthreshold,remove_edge,show=False):
        boxs = []
        rawboxs={}
        for p in range(particles_metrics.shape[0]):
            std=particles_metrics[p,0]
            if std >= stdthreshold and (std>particles_metrics[p,1] or (not remove_edge)):
                if std not in rawboxs:
                    rawboxs[std]=[]
                rawboxs[std].append([boxes[p][0],boxes[p][1],boxes[p][2]])
        rawboxs=[rawboxs[k] for k in sorted(rawboxs.keys())]
        for b in rawboxs:
            boxs+=b

        f = open('boxs.txt', 'w')
        f.writelines([' '.join(str(boxs[i])[1:-1].split(', '))+'\n' for i in range(len(boxs))])
        f.close()
        circle = int(self.radius / self.pixelsize / self.auto_binning)
        # Saving coordinates in spk format and swapping Y-Z
        run_shell_command(get_imod_path()+'/bin/point2model boxs.txt '+self.name+'.mod -sphere %s' % circle,  verbose=False)
        run_shell_command(get_imod_path()+'/bin/imodtrans -Y -T ' +self.name+ '.mod ' + self.name + ".spk", verbose=False )
        logger.info(f"\t{len(boxs):,} particles detected")

        if show:
            run_shell_command(get_imod_path()+"/bin/imod -xyz -Y bp.mrc "+self.name+".spk")
            self.show_particles(boxs,times=3)

    @timer.Timer(
        "minima_extract", text="\tTotal time elapsed (picker minima_extract): {}", logger=logger.info
    )
    def minima_extract(self,radius_times=4,inhibit=False):
        locality = int(radius_times*self.radius / self.pixelsize / self.auto_binning)
        if not inhibit:
            minimas = (
                self.lowres == scipy.ndimage.minimum_filter(self.lowres, locality)
            ).nonzero()
            points=[]
            for i in range(minimas[0].shape[0]):
                points.append([minimas[0][i],minimas[1][i],minimas[2][i]])
            return points
        locality2 = int(radius_times*self.radius/2 / self.pixelsize / self.auto_binning)
        minimas = (
            self.lowres == scipy.ndimage.minimum_filter(self.lowres, locality2)
        ).nonzero()
        rawpoints={}
        for i in range(minimas[0].shape[0]):
            x,y,z=minimas[0][i],minimas[1][i],minimas[2][i]
            v=self.lowres[x,y,z]
            if v not in rawpoints:
                rawpoints[v]=[]
            rawpoints[v].append([[x,y,z],False])
        editmax=self.lowres.max()
        edited=self.lowres+(1-(self.lowres == scipy.ndimage.minimum_filter(self.lowres, locality2)))*(editmax-self.lowres)
        inhibited=1
        while inhibited>0:
            inhibited=0
            minf=scipy.ndimage.minimum_filter(edited, locality)

            editmax=edited.max()

            for k in rawpoints.keys():
                for i in range(len(rawpoints[k])):
                    p=rawpoints[k][i]
                    minimum=minf[p[0][0],p[0][1],p[0][2]]
                    if minimum==k:
                        p[1]=False
                    elif minimum in rawpoints:
                        p[1]=minimum
                        inhibited+=1
                    else:
                        p[1]=True
                        inhibited+=1
                        x,y,z=p[0][0]-int(locality/2),p[0][1]-int(locality/2),p[0][2]-int(locality/2)
                        a,b,c=x+locality,y+locality,z+locality
                        x=0 if x<0 else x
                        y=0 if y<0 else y
                        z=0 if z<0 else z
                        a=edited.shape[0] if a>edited.shape[0] else a
                        b=edited.shape[1] if b>edited.shape[1] else b
                        c=edited.shape[2] if c>edited.shape[2] else c
                        edited[x:a,y:b,z:c]+=(edited[x:a,y:b,z:c]==minimum)*(editmax-edited[x:a,y:b,z:c])

            deleting={}
            for k in rawpoints.keys():
                for i in range(len(rawpoints[k])):
                    p=rawpoints[k][i]
                    if type(p[1])!=bool:
                        d=False
                        for j in rawpoints[p[1]]:
                            if type(j[1])==bool and j[1]==False:
                                d=True
                                break
                        if d:
                            if k not in deleting:
                                deleting[k]=[]
                            deleting[k].append(i)


            for k in deleting.keys():
                for i in range(len(deleting[k])-1,-1,-1):
                    point=rawpoints[k][deleting[k][i]][0]
                    edited[point[0],point[1],point[2]]=editmax
                    rawpoints[k].pop(deleting[k][i])
                if len(rawpoints[k])==0:
                    del rawpoints[k]
        points=[]
        for k in rawpoints.keys():
            for p in rawpoints[k]:
                points.append(p[0])
        return points

    def show_particles(self,boxs,times=3):
        outboxs=[]
        for b in boxs:
            x = b[2]
            y = b[1]
            z = b[0]
            inside = (
                x- times*self.tilesize / 2 >= 0
                and x < self.rec.shape[0] - times*self.tilesize/2 + 1
                and y- times*self.tilesize / 2 >= 0
                and y < self.rec.shape[1] - times*self.tilesize/2 + 1
                and z- times*self.tilesize / 2 >= 0
                and z < self.rec.shape[2] - times*self.tilesize/2 + 1
            )
            if inside:
                outboxs.append([z,y,x])
        outmrc = extract(self.rec, outboxs,self.tilesize*times)
        mrc.write(outmrc.sum(axis=2), "particles.mrc")
        run_shell_command(get_imod_path()+"/bin/imod -xyz particles.mrc")

@timer.Timer(
    "pick", text="Total time elapsed (picker): {}", logger=logger.info
)


def pick(name,radius=100,pixelsize=2.1,auto_binning = 12,contract_times=1,gaussian=False,sigma=15,stdtimes_cont=3.5,min_size=125,dilation=100,radius_times=4,inhibit=False,detection_width=128,stdtimes_filt=1,remove_edge=False,show=False):

    p=Picker(name,radius=radius,pixelsize=pixelsize,auto_binning = auto_binning,show=show)
    area=p.getcont(contract_times=contract_times,gaussian=gaussian,sigma=sigma,stdtimes=stdtimes_cont,min_size=min_size,dilation=dilation,show=show)

    boxes,raw_particles=p.detect(area,contract_times=contract_times,radius_times=radius_times,inhibit=False,detection_width=detection_width,show=show)
    particles_metrics,stdthreshold=p.prefilt(raw_particles,stdtimes=stdtimes_filt)
    if inhibit:
        boxes,raw_particles=p.detect(area,contract_times=contract_times,radius_times=radius_times,inhibit=True,detection_width=detection_width,show=show)
        particles_metrics,_=p.prefilt(raw_particles,stdtimes=stdtimes_filt)
    p.filt(boxes,particles_metrics,stdthreshold,remove_edge,show)
