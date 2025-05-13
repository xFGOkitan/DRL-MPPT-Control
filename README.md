# DRL MPPT Control
 Hello, this is my first major project using Deep Reinforcement Learning for MPPT Control in a solar panel system.
This project is part of my final university coursework, where I am exploring the use of DRL for Maximum Power Point Tracking (MPPT) in solar panels. I chose this approach because of AI’s growing potential and its promising applications in optimizing energy efficiency, as seen in various research studies.

Currently, I am fine-tuning the reward system to ensure stricter reinforcement, as well as troubleshooting an issue with one of the final output graphs not displaying correctly.

# 2nd Update
Hello again. This is my 2nd update for my DRL MPPT Project. I was spending time making a P&O (Perturb and Observe) version to compare my DRL model. The P&O method was done in MATLAB as a recommendation from one of my labmates, so I also began learning how to code in MATLAB. If you run the code, you will see that my DRL method can maintain the theoretical maximum MPPT value even through rapid changes in irradiance. The P&O method is also running properly and can keep up with the DRL model, but the main issue with P&O is that there are oscillations (small but present). These oscillations lower the efficiency of MPP tracking, which results in power output losses. 

For future updates to my DRL model, I am hoping to test the model using a randomly changing irradiance (not rapidly changing, one more in tune with the environment), and I think I should be able to finish that in a month or so. 

Thank you for taking the time to learn about my project, and any recommendations are welcome.
