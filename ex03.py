# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:37:00 2019

@author: Anthony
"""

from experiment_shell import Experiment

def main():
    """
    """
    ex = Experiment()
    
    print("\nCars Data:")
    for k in (1,3,5,7,11,15,23,29):
        print("\nfor k of",k)
        ex.run("2",k,"2",0.3,10)
    
    print("\n\nAuto-MPG Data:")
    for k in (1,3,5,7,11,15,23,29):
        print("\nfor k of",k)
        ex.run("3",k,"3",0.3,10)
    
    print("\n\nStudent (Math) Data:")
    for k in (1,3,5,7,11,15,23,29):
        print("\nfor k of",k)
        ex.run("3",k,"4",0.3,10)
    
    print("\n\nStudent (Math) Data (Discrete):")
    for k in (1,3,5,7,11,15,23,29):
        print("\nfor k of",k)
        ex.run("3",k,"5",0.3,10)
    pass
    

if __name__ == "__main__":
    main()