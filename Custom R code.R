############R code for this study

rm(list=ls())

library(pROC)
library(rms)
library(data.table)
library(rmda)
library("survival")
library("survminer")

#change work directory
setwd("**path**")

#load data
nomotrain = read.csv(file = '**path**') ######internal set
nomotest = read.csv(file = '**path**') ######external set
nomoall=read.csv(file = '**path**')
ddist <- datadist(nomotrain); options(datadist='ddist')

###1. for logistics regression
logi=glm(AP~DL_classifier_100+PB_ISUP+cT+PSA+PIRADS,  data=nomotrain, family=binomial(link="logit"))

logi2=glm(AP~DL_classifier_dichotomous+PB_ISUP+cT+PSA+PIRADS,  data=nomotrain, family=binomial(link="logit"))

LogisFormat<-function(fit){
  #P value
  p<-summary(fit)$coefficients[,4]
  #wald Value
  wald<-summary(fit)$coefficients[,3]^2
  #B Value
  valueB<-coef(fit)
  #OR Value
  valueOR<-exp(coef(fit))
  #OR 95%CI
  confitOR<-exp(confint(fit))
  data.frame(
    B=round(valueB,3),
    Wald=round(wald,3),
    OR_with_CI=paste(round(valueOR,3),"(",
                     round(confitOR[,1],3),"~",round(confitOR[,2],3),")",sep=""),
    P=format.pval(p,digits = 3,eps=0.001)
  )
}

LogisFormat(logi) ####logistics output


###2. for nomogram
fitDLnomogram <- lrm(AP ~ DL_classifier+PB_ISUP+cT, x=T, y=T, data=nomotrain,penalty = 1)

nom <- nomogram(fitDLnomogram, lp=FALSE, fun=function(x)1/(1+exp(-x)),fun.at=c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),funlabel="Predictived Value")
plot(nom, xfrac=.22)

cal <- calibrate(fitDLnomogram,B=1000,  predy=seq(.2, .8, length=60))
plot(cal, xlab=c("Predicted probability (with AP)"))


###3. for ROC curves 
##forResNet50 and NAFNet comparisons
rocDL_classifier <-roc(AP~DL_classifier, x=T, y=T, data=nomotest,penalty = 1,ci=T)
rocResNet <- roc(AP~ResNet, x=T, y=T, data=nomotest,penalty = 1,ci=T) 

plot.roc(rocDL_classifier, legacy.axes=T, identity.col="black", identity.lwd=5, print.auc=FALSE, print.thres=FALSE, col="red", lwd=5, cex.axis=1.7)
plot.roc(rocResNet, add=TRUE, auc.polygon=FALSE, col="grey", lwd=5)

##for NAFNet and other models comparions
fitDLnomogram <- lrm(AP ~ DL_classifier+PB_ISUP+cT, x=T, y=T, data=nomotrain,penalty = 1)
nomopredDLnomogram = predict(fitDLnomogram,nomotest,type="fitted.ind")
rocDLnomogram <- roc(nomotest$AP, nomopredDLnomogram, ci=T)

rocDL_classifier <-roc(AP~DL_classifier, x=T, y=T, data=nomotest,penalty = 1,ci=T)

rocPIRADS <- roc(AP~PIRADS, x=T, y=T, data=nomotest,penalty = 1,ci=T)

rocCAPRA<- roc(AP~CAPRA_3, x=T, y=T, data=nomotest,penalty = 1,ci=T) 

#DLnomogram,blue; DL_classifier, red; PIRADS, green; CAPRA, purple; 

plot.roc(rocDLnomogram, legacy.axes=T, identity.col="black", identity.lwd=4, print.auc=FALSE, print.thres=FALSE, col="blue", lwd=3, cex.axis=1.2)
plot.roc(rocDL_classifier, add=TRUE, auc.polygon=FALSE, col="red", lwd=3)
plot.roc(rocPIRADS, add=TRUE, auc.polygon=FALSE, col="green", lwd=3)
plot.roc(rocCAPRA, add=TRUE, auc.polygon=FALSE, col="purple", lwd=3)
auc(rocDLnomogram) 
ci.auc(rocDLnomogram) 
#and other models


###ROC compare
roc.test(rocDLnomogram,rocCAPRA) 
#and other models

###4. for DCA curves 
DCA<- read.csv(file = '**path**')
DCADLnomogram<- decision_curve(AP ~ DL_classifier+cT+PB_ISUP,  data = DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCADL_classifier <- decision_curve(AP ~ DL_classifier,  data = DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCAPIRADS <- decision_curve(AP ~ PIRADS,  data = DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCACAPRA <- decision_curve(AP ~ CAPRA_3,  data =DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort')) 

DCAResNet <- decision_curve(AP ~ ResNet50,  data =DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort')) 

List<- list(DCADLnomogram, DCADL_classifier, DCAPIRADS, DCACAPRA, DCAResNet)

plot_decision_curve(List ,curve.names=c("DLnomogram", "DL_classifier", "PIRADS", "CAPRA", "ResNet50"), col=c("blue","red","green","purple","orange"), ylim=c(-0.18,0.6), legend.position="none", cost.benefit.axis =F, confidence.intervals =FALSE, standardize = FALSE, lwd=c(3,3,3,3,3,5), cex.axis=1.2)

###5. for KM curves
###### Threshold
plot.roc(rocDLnomogram, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, print.thres.pattern=, col="blue", lwd=2 ) 
#and other models

###### Saving predictions for further analysis. 
dfDL_classifier <- data.frame(nomotest$Graphic_ID,nomotest$AP,nomotest$DL_classifier)
dfDLnomogram <- data.frame(nomotest$Graphic_ID,nomotest$AP,nomopredDLnomogram)

write.csv(dfDL_classifier,'**path**', row.names = TRUE)
write.csv(dfDLnomogram,'**path**', row.names = TRUE)
#and other models

########### KM 
KM <- read.csv(file = '**path**')

KMDLnomogram<- survfit(Surv(Time_to_BCR, BCR) ~ DLnomogram_2, data = KM)
ggsurvplot(KMDLnomogram, data= KM, pval = T, legend.labs =c("DL-nomogram 0","DL-nomogram 1"), conf.int = TRUE, risk.table = TRUE, risk.table.col = "strata", linetype = "strata", surv.median.line = "hv", ggtheme = theme_bw(), palette = c("#E7B800", "#2E9FDF"), break.time.by=12,xlim=c(0,55),xlab="Months after surgery", ylab="BCR-free survival",tables.y.text=F) 

KMDL_classifier<- survfit(Surv(Time_to_BCR, BCR) ~ DL_classifier_2, data = KM)
ggsurvplot(KMDL_classifier, data= KM, pval = T, legend.labs =c("DL-classifier 0","DL-classifier 1"), conf.int = TRUE, risk.table = TRUE, risk.table.col = "strata", linetype = "strata", surv.median.line = "hv", ggtheme = theme_bw(), palette = c("#E7B800", "#2E9FDF"), break.time.by=12,xlim=c(0,55),xlab="Months after surgery", ylab="BCR-free survival",tables.y.text=F) 

KMPIRADS<- survfit(Surv(Time_to_BCR, BCR) ~ PIRADS_2, data = KM)
ggsurvplot(KMPIRADS, data= KM, pval = T, legend.labs =c("PI-RADS<5","PI-RADS≥5"), conf.int = TRUE, risk.table = TRUE, risk.table.col = "strata", linetype = "strata", surv.median.line = "hv", ggtheme = theme_bw(), palette = c("#E7B800", "#2E9FDF"), break.time.by=12,xlim=c(0,55), xlab="Months after surgery",ylab="BCR-free survival",tables.y.text=F) 

KMCAPRA<- survfit(Surv(Time_to_BCR, BCR) ~ CAPRA_3_2, data = KM)
ggsurvplot(KMCAPRA, data= KM, pval = T, legend.labs =c("CAPRA<6","CAPRA≥6"), conf.int = TRUE, risk.table = TRUE, risk.table.col = "strata", linetype = "strata", surv.median.line = "hv", ggtheme = theme_bw(), palette = c("#E7B800", "#2E9FDF"), break.time.by=12,xlim=c(0,55),xlab="Months after surgery", ylab="BCR-free survival",tables.y.text=F) 


######c-index built cox model
coxDL_classifier<- coxph(Surv(Time_to_BCR, BCR) ~ DL_classifier, data = KM) 
summary(coxDL_classifier)

coxDLnomogram<- coxph(Surv(Time_to_BCR, BCR) ~ DL_classifier+cT+PB_ISUP, data = KM)
summary(coxDLnomogram) 

coxPIRADS <- coxph(Surv(Time_to_BCR, BCR) ~ PIRADS, data = KM) #0.630 se=0.03
summary(coxPIRADS)

coxCAPRA_3 <- coxph(Surv(Time_to_BCR, BCR) ~ CAPRA_3, data = KM)
summary(coxCAPRA_3) 

#########comparing c-indices
anova(coxCAPRA_3, coxDL_classifier) 
#and other moedels


###6. Supplementary
#######For confusion matrices
library(caret)
lvs <- c("with AP", "withou AP")
Truth <- factor(rep(lvs, times = c(XX, XX)),
                levels = rev(lvs))

Pred <- factor(
  c(
    rep(lvs, times = c(XX, XX)),
    rep(lvs, times = c(XX, XX))),
  levels = rev(lvs))

Xtab <- table(Pred, Truth)

confusionMatrix(Xtab, positive="with AP")


#######For comparing different nomogram based on DL classifier
fitDLnomogram <- lrm(AP ~ DL_classifier+PB_ISUP+cT, x=T, y=T, data=nomotrain,penalty = 1)
nomopredDLnomogram = predict(fitDLnomogram,nomotest,type="fitted.ind")
rocDLnomogram <- roc(nomotest$AP, nomopredDLnomogram, ci=T)
auc(rocDLnomogram) 
rocDLnomogram 

# Had PSA
fitDLnomogram_PSA <- lrm(AP ~ DL_classifier+cT+PB_ISUP+PSA_2, x=T, y=T, data=nomotrain,penalty = 1)
nomopredDLnomogram_PSA = predict(fitDLnomogram_PSA,nomotest,type="fitted.ind")
rocDLnomogram_PSA =roc(nomotest$AP, nomopredDLnomogram_PSA,ci=T)
rocDLnomogram_PSA
auc(rocDLnomogram_PSA) 

#Had PI-RADS
fitDLnomogram_PI <- lrm(AP ~ DL_classifier+cT+PB_ISUP+PIRADS, x=T, y=T, data=nomotrain,penalty = 1)
nomopredDLnomogram_PI = predict(fitDLnomogram_PI,nomotest,type="fitted.ind")
rocDLnomogram_PI <- roc(nomotest$AP, nomopredDLnomogram_PI, ci=T)
rocDLnomogram_PI
auc(rocDLnomogram_PI) 

#Had PSA and PI-RADS
fitDLnomogram_P2 <- lrm(AP ~ DL_classifier+cT+PB_ISUP+PIRADS+PSA_2, x=T, y=T, data=nomotrain,penalty = 1)
nomopredDLnomogram_P2 = predict(fitDLnomogram_P2,nomotest,type="fitted.ind")
rocDLnomogram_P2 <- roc(nomotest$AP, nomopredDLnomogram_P2, ci=T)
rocDLnomogram_P2
auc(rocDLnomogram_P2) 

#comparing
roc.test(rocDLnomogram,rocDLnomogram_PI) 
roc.test(rocDLnomogram,rocDLnomogram_PSA) 
roc.test(rocDLnomogram,rocDLnomogram_P2) 

################nomogram
nomDLnomogram_PSA <- nomogram(fitDLnomogram_PSA, lp=FALSE, fun=function(x)1/(1+exp(-x)),fun.at=c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),funlabel="Predictived Value")
plot(nomDLnomogram_PSA, xfrac=.22)

calDLnomogram_PSA <- calibrate(fitDLnomogram_PSA,B=1000,  predy=seq(.2, .8, length=60))
plot(calDLnomogram_PSA, xlab=c("Predicted probability (with AP)")) 

nomDLnomogram_PI <- nomogram(fitDLnomogram_PI, lp=FALSE, fun=function(x)1/(1+exp(-x)),fun.at=c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),funlabel="Predictived Value")
plot(nomDLnomogram_PI, xfrac=.22)

calDLnomogram_PI <- calibrate(fitDLnomogram_PI,B=1000,  predy=seq(.2, .8, length=60))
plot(calDLnomogram_PI, xlab=c("Predicted probability (with AP)")) 


nomDLnomogram_P2 <- nomogram(fitDLnomogram_P2, lp=FALSE, fun=function(x)1/(1+exp(-x)),fun.at=c(.001,.01,.05,seq(.1,.9,by=.1),.95,.99,.999),funlabel="Predictived Value")
plot(nomDLnomogram_P2, xfrac=.22)

calDLnomogram_P2 <- calibrate(fitDLnomogram_P2,B=1000,  predy=seq(.2, .8, length=60))
plot(calDLnomogram_P2, xlab=c("Predicted probability (with AP)")) #699*483 ratio




###############ROC DLnomogram,blue; DL_classifier, red; PIRADS, green; CAPRA, purple; 895*794 
plot.roc(rocDLnomogram, legacy.axes=T, identity.col="black", identity.lwd=4, print.auc=FALSE, print.thres=FALSE, col="blue", lwd=2, cex.axis=1.2)
plot.roc(rocDLnomogram_PSA, add=TRUE, auc.polygon=FALSE, col="red", lwd=2)
plot.roc(rocDLnomogram_PI, add=TRUE, auc.polygon=FALSE, col="green", lwd=2)
plot.roc(rocDLnomogram_P2, add=TRUE, auc.polygon=FALSE, col="purple", lwd=2)

########DCA CURVE
DCA<- read.csv(file = '**path**')
DCADLnomogram<- decision_curve(AP ~ DL_classifier+cT+PB_ISUP,  data = DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCADLnomogram_PSA <- decision_curve(AP ~ DL_classifier+cT+PB_ISUP+PSA_2,  data = DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCADLnomogram_PI <- decision_curve(AP ~ DL_classifier+cT+PB_ISUP+PIRADS,  data = DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCADLnomogram_P2 <- decision_curve(AP  ~ DL_classifier+cT+PB_ISUP+PIRADS+PSA_2,  data =DCA, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort')) 

List<- list(DCADLnomogram, DCADLnomogram_PSA, DCADLnomogram_PI, DCADLnomogram_P2)

plot_decision_curve(List ,curve.names=c("DLnomogram", "DLnomogram+PSA", "DLnomogram+PIRADS", "DLnomogram+PSA+PIRADS"), col=c("blue","red","green","purple"), ylim=c(-0.18,0.6), legend.position="none", cost.benefit.axis =F, confidence.intervals =FALSE, standardize = FALSE, lwd=c(3,3,3,3,4))




####Specificity Sensitivity
plot.roc(rocDLnomogram, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, print.thres.pattern=, col="blue", lwd=2 ) 
plot.roc(rocDLnomogram_PSA, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, col="blue", lwd=2 ) 
plot.roc(rocDLnomogram_PI, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, col="blue", lwd=2 ) 
plot.roc(rocDLnomogram_P2, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, col="blue", lwd=2 ) 

# Saving predictions for further analysis. 
dfDLnomogram_P2 <- data.frame(nomotest$Graphic_ID,nomotest$AP,nomopredDLnomogram_P2)
dfDLnomogram_PI <- data.frame(nomotest$Graphic_ID,nomotest$AP,nomopredDLnomogram_PI)
dfDLnomogram_PSA <- data.frame(nomotest$Graphic_ID,nomotest$AP,nomopredDLnomogram_PSA)

write.csv(dfDLnomogram_PSA,'**path**', row.names = TRUE)
write.csv(dfDLnomogram_PI,'**path**', row.names = TRUE)
write.csv(dfDLnomogram_P2,'**path**', row.names = TRUE)

######For predicting post-surgery ISUP
###
ISUPlogi=glm(ISUP_2~DL_classifier_100+cT+PSA_2+PIRADS,  data=nomoall, family=binomial(link="logit"))
LogisFormat(ISUPlogi)

fitDLPT <- lrm(ISUP_2 ~ DL_classifier+cT+PSA_2, x=T, y=T, data=nomoall,penalty = 1)
nomopredDLPT = predict(fitDLPT,nomoall,type="fitted.ind")
rocDLPT <- roc(nomoall$ISUP_2, nomopredDLPT,ci=T)
rocDLPT 

rocDL_classifierISUP <-roc(ISUP_2~DL_classifier, x=T, y=T, data=nomoall,penalty = 1,ci=T) 
rocDL_classifierISUP 

rocPIRADSISUP <- roc(ISUP_2~PIRADS, x=T, y=T, data=nomoall,penalty = 1,ci=T)
rocPIRADSISUP 

rocPBCGISUP <-roc(ISUP_2~PBCG_High, x=T, y=T, data=nomoall,penalty = 1, ci=T)
rocPBCGISUP 


roc.test(rocPBCGISUP,rocDLPT) 
#and other models


#########ROC CURVE
######DLPT,blue; DL_classifierISUP, red; PIRADSISUP, green; PBCGISUP, purple;
plot.roc(rocDLPT, legacy.axes=T, identity.col="black", identity.lwd=4, print.auc=FALSE, print.thres=FALSE, col="blue", lwd=3, cex.axis=1.2)
plot.roc(rocDL_classifierISUP, add=TRUE, auc.polygon=FALSE, col="red", lwd=3)
plot.roc(rocPIRADSISUP, add=TRUE, auc.polygon=FALSE, col="green", lwd=3)
plot.roc(rocPBCGISUP, add=TRUE, auc.polygon=FALSE, col="purple", lwd=3)

########DCA CURVE 
DCADLPT<- decision_curve(ISUP_2 ~ DL_classifier+cT+PSA_2,  data = nomoall, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCADL_classifierISUP <- decision_curve(ISUP_2 ~ DL_classifier,  data = nomoall, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCAPIRADSISUP <- decision_curve(ISUP_2 ~ PIRADS,  data = nomoall, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort'))

DCAPBCGISUP<- decision_curve(ISUP_2 ~PBCG_High,  data = nomoall, family = binomial(link ='logit'), thresholds = seq(0,1, by = 0.01), confidence.intervals= 0.95, bootstraps = 1000, study.design = c('cohort')) 

List<- list(DCADLPT, DCADL_classifierISUP, DCAPIRADSISUP, DCAPBCGISUP)

plot_decision_curve(List ,curve.names=c("DLPT", "DL_classifier", "PIRADS", "PBCG"), col=c("blue","red","green","purple"), legend.position="none", cost.benefit.axis =F, confidence.intervals =FALSE, standardize = FALSE, lwd=c(3,3,3,3,4))


#######Specificity Sensitivity
plot.roc(rocDLPT, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T,  col="blue", lwd=2 ) 
plot.roc(rocPIRADSISUP, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, col="blue", lwd=2 )  
plot.roc(rocDL_classifierISUP, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, col="blue", lwd=2 ) 
plot.roc(rocPBCGISUP, legacy.axes=T, identity.col="black", identity.lwd=2, print.auc=FALSE, print.thres=T, col="blue", lwd=2 )  

# Saving predictions for further analysis. 
dfDLPT <- data.frame(nomoall$Graphic_ID,nomoall$ISUP_2,nomopredDLPT)
dfPIRADSISUP <- data.frame(nomoall$Graphic_ID,nomoall$ISUP_2,nomoall$PIRADS)
dfDL_classifierISUP <- data.frame(nomoall$Graphic_ID,nomoall$ISUP_2,nomoall$DL_classifier)
dfPBCGISUP <- data.frame(nomoall$Graphic_ID,nomoall$ISUP_2,nomoall$PBCG_High)

write.csv(dfDLPT,'**path**', row.names = TRUE)
write.csv(dfPIRADSISUP,'**path**', row.names = TRUE)
write.csv(dfDL_classifierISUP,'**path**', row.names = TRUE)
write.csv(dfPBCGISUP,'**path**', row.names = TRUE)
