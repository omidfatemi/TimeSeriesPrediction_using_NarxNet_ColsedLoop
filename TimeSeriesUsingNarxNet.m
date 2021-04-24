clear all
%%
load magdata
y = con2seq(y);
u = con2seq(u);
TimeSteps=length(y);
numTrSamples  = round(.5*TimeSteps); %devide data for training and validation
d1 = [1:2]; %Netwrok Number of Delays
d2 = [1:2];
%
WindowLength=d2(end);
extInputDelays=d1(end);
numInputs =1;
% NarxNet  
narx_net = narxnet(d1,d2,10);
narx_net.divideFcn = '';    %no early stopping
narx_net.trainParam.min_grad = 1e-10;
narx_net.inputs{1}.processFcns ={ 'removeconstantrows'};
narx_net.inputs{2}.processFcns ={ 'removeconstantrows'};
[p,Pi,Ai,t] = preparets(narx_net,u(1:numTrSamples),{},y(1:numTrSamples)); %no normalization is applied 
narx_net = train(narx_net,p,t,Pi);
view(narx_net)
narxnetClosedLoop = closeloop(narx_net);
%% validation in closed-loop form
wb = getwb(narxnetClosedLoop);
[b,IW,LW] =separatewb(narxnetClosedLoop,wb);
view(narx_net)
%% 
dataX = zeros(TimeSteps,WindowLength);
dataInputX =zeros(TimeSteps,extInputDelays);
yPred= zeros(size(y));
yPred(1:numTrSamples-WindowLength)=cell2mat(y(1:numTrSamples-WindowLength));
%%
for j= numTrSamples-WindowLength+1 :TimeSteps- WindowLength
    for idx =1: WindowLength
%         dataX(j,idx) =y{1,j+idx-1};
        dataX(j,idx) =yPred(j-idx);
    end

    for idx =1: extInputDelays
        dataInputX(j,idx) = u{1,j-idx};
    end    
    L1 = tansig(IW{1, 1}*dataInputX(j,:)'+LW{1, 2}*dataX(j,:)'+b{1,1});
    yPred(j)= purelin(LW{2,1}*L1+b{2,1});    
    disp(j)
end
%%
e = (yPred)-cell2mat(y);
figure,plot(e)
%%
k =numTrSamples-WindowLength+1 :TimeSteps- WindowLength;
figure, plot(k,yPred(k),k,cell2mat(y(k)))
legend('yPred','y')
%%
err=immse(yPred,cella2mat(y))
%%