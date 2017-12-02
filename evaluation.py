#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 15:19:54 2017

@author: austinlee
"""

from collections import Counter

class Eval:
    def __init__(self, gold, pred):
        assert len(gold)==len(pred)
        self.gold = gold
        self.pred = pred

    def print_matrix(self, matrix):
        	s = [[str(e) for e in row] for row in matrix]
        	lens = [max(map(len, col)) for col in zip(*s)]
        	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        	table = [fmt.format(*row) for row in s]
        	print ('\n'.join(table))
    
    def accuracy(self):
        numer = sum(1 for p,g in zip(self.pred,self.gold) if p==g)
        
        # Build confusion matrix
        confmatrx = [[0 for x in range(4)] for y in range(4)]
        row=0
        col=0
        for p,g in zip(self.pred, self.gold):
            if p=="AGAINST":
                row=0
            elif p=="FAVOR":
                row=1
            elif p=="NONE":
                row=2
            
            if g=="AGAINST":
                col=0
            elif g=="FAVOR":
                col=1
            elif g=="NONE":
                col=2
            
            confmatrx[row][col]+=1
        for r in range(0,3):
            rowSum=0
            for c in range(0,3):
                rowSum+=confmatrx[r][c]
            confmatrx[r][3]=rowSum
        for c in range(0,3):
            colSum=0
            for r in range(0,3):
                colSum+=confmatrx[r][c]
            confmatrx[3][c]=colSum
        mtxSum = 0
        for g in range(0,3):
            mtxSum+=confmatrx[3][g]
        confmatrx[3][3]=mtxSum

        # Print confusion matrix
        self.print_matrix(confmatrx)
        
        # Precision, recall, and F1
        against_preds = 0
        favor_preds = 0
        none_preds = 0
        
        against_golds = 0
        favor_golds = 0
        none_golds = 0
        
        against_pgy = 0
        favor_pgy = 0
        none_pgy = 0
    
        for p,g in zip(self.pred, self.gold):
            if p==g and p=="AGAINST":
                against_pgy+=1
            elif p==g and p=="FAVOR":
                favor_pgy+=1
            elif p==g and p=="NONE":
                none_pgy+=1
                
            if g=="AGAINST":
                against_golds+=1
            elif g=="FAVOR":
                favor_golds+=1
            elif g=="NONE":
                none_golds+=1
           
            if p=="AGAINST":
                against_preds+=1
            elif p=="FAVOR":
                favor_preds+=1
            elif p=="NONE":
                none_preds+=1
        
        print("The precision of the classifier for each stance is as follows: ")
        print("AGAINST: " + str(against_pgy/against_preds))
        print("FAVOR: " + str(favor_pgy/favor_preds))
        print("NONE: " + str(none_pgy/none_preds))
        
        print("The recall of the classifier for each stance is as follows: ")
        print("AGAINSt: " + str(against_pgy/against_golds))
        print("FAVOR: " + str(favor_pgy/favor_golds))
        print("NONE: " + str(none_pgy/none_golds))
        
        print("The F1 score of the classifier for each stance is as follows: ")
        print("AGAINST: " + str((2*(against_pgy/against_preds)*(against_pgy/against_golds))/((against_pgy/against_preds)+(against_pgy/against_golds))))
        print("FAVOR: " + str((2*(favor_pgy/favor_preds)*(favor_pgy/favor_golds))/((favor_pgy/favor_preds)+(favor_pgy/favor_golds))))
        print("NONE: " + str((2*(none_pgy/none_preds)*(none_pgy/none_golds))/((none_pgy/none_preds)+(none_pgy/none_golds))))
        
        return numer / len(self.gold)