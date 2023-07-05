# plots the jacobian and minimum-maximum forces over the workspace

import fwd_kinematics
import inv_kinematics

#clear
#clc
#close all
#set(0,'defaultfigurewindowstyle','docked')

# plot the linkage at each step?
plot_linkage = false

# force angles
angle_number = 20
angle_range = 2*np.pi/angle_number

# max current per motor
kt = 0.0234

# based on stall torue (max peak)
Tmax = 0.129
Imax = Tmax/kt

# based on maximum continuous
# Imax = 1.2; #A
# Tmax = kt*Imax;

# capstan mechanial advantage ratio
ratio = 0.07303 / (0.0131/2)

# link lengths
a1 = .25
a2 = .25
a3 = a2
a4 = a1
a5 = .1

# origin
x1 = 0
y1 = 0

# the rectangle
xcenter = -(x1+a5)/2
ycenter = .34
w = .2794
h = .2159


# define how many points to solve for
number = 10
xpoints = np.linspace(xcenter + w/2, xcenter - w/2, number)
ypoints = np.linspace(ycenter + h/2, ycenter - h/2, number)

# preallocate
condition = np.zeros(length(xpoints),length(ypoints))
theta     = np.zeros(2,length(xpoints))
mean_force = np.zeros(length(xpoints),length(ypoints))
min_force = np.zeros(length(xpoints),length(ypoints))

# at every point
for i in range(len(xpoints)):
    for j in range(len(ypoints)):
        
        #find joint angles
        x3 = xpoints[i].item()
        y3 = ypoints[j].item()
        
        theta = inv_kinematics.inv_kinematics(a1,a2,a3,a4,a5,x3,y3)
        t1 = theta[0].item()
        t5 = theta[1].item()
        
        # jacobian
        J = fwd_kinematics.fwd_kinematics(a1,a2,a3,a4,a5,t1,t5,plot_linkage)
        
        Jinv = np.linalg.inv(J)
        
        # condition of jacobian
        condition(i,j) = cond(J);
        
        
        #solve torque needed for 10 force directions evenly spaced
        # increase the magnitue of the force until T1 or T2 > Tmax
        # report the largest and the smallest
        
        
        
        # for each point
        for k = 1:angle_number;
            mag = 0;
            max_force_reached = false;
            #while we have not exceeded the maximum toruqe
            while(~max_force_reached)
                #find the force in a particular direction
                f(:,k) = mag*[cos(angle_range) -sin(k*angle_range); sin(k*angle_range) cos(k*angle_range)]*[1;0];
                # compute the needed motor torque
                torque = J'*f(:,k);
                # if we have  exceeded spec (90% of max current)
                if( abs(torque(1)) >= (Tmax)*ratio || abs(torque(2)) >= (Tmax)*ratio )
                    #then we have reached the max force in that direction
                    max_force_reached = true;
                else
                    #otherwise increment the magnitude
                    mag = mag+.01;
                end
            end
            max_dir_force(i,j,k) = mag;
        end
        mean_force(i,j) = mean(max_dir_force(i,j,:));
        min_force(i,j) = min(max_dir_force(i,j,:));
    end
end


#FIGURE 1
# figure set up
xlabel('-x');ylabel('-y')
set(gcf,'color','white');
axis equal
axis([-.2 .35 -.45 0.03])
grid on
hold on

#plot one linkage configuration
t1iso = .8719;
t5iso = 2.2697;
J = fwd_kinematics(a1,a2,a3,a4,a5,t1iso,t5iso,true);

# plot the rectangle
rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])

#plot the condition number contour
v = [1:.5:4];
[c handle] = contour(-xpoints,-ypoints,condition', v);
clabel(c,handle,v)



#FIGURE 2
# figure set up
figure
hold on
xlabel('-x [m]');ylabel('-y [m]')
set(gcf,'color','white');
axis equal
axis([-.2 .35 -.45 0.03])
grid on

#plot one linkage configuration
t1iso = .8719;
t5iso = 2.2697;
J = fwd_kinematics(a1,a2,a3,a4,a5,t1iso,t5iso,true);

# plot the rectangle
rectangle('Position',[-(xcenter+w/2) -(ycenter+h/2) w h])

#plot the min force contour
v = 0:.5:5;
[c handle] = contour(-xpoints,-ypoints,min_force', v);
clabel(c,handle,v)








