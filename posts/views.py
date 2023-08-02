from django.shortcuts import render, get_object_or_404, render_to_response

import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd

from .models import Post

from posts.forms import stocksFORM, InputForm, ReportForm, CandleForm, NewsletterForm
from posts.models import Stocks, Input, Report, Candle, newsletter

import pandas_datareader.data as web
import numpy as np
from datetime import datetime

from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.template.loader import get_template

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
#import pandas_datareader as web-- If you are using python 3.5
#from pandas_datareader import data as web
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
#import pandas_datareader.data as web
import random

from sklearn import covariance, cluster
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn import ensemble

from django.utils.formats import localize

# Create your views here.


def home(request):
    posts = Post.objects.order_by('-pub_date')
    return render (request, 'posts/home_page.html',{'posts':posts})


def post_details(request, post_id):
    post = get_object_or_404(Post, pk=post_id )
    return render(request, 'posts/post_detail.html',{'post':post})

# import plotly.tools as tls
# tls.set_credentials_file(username='chemalle', api_key='3g2DEUppR01VxRz3P8NW')

# #username = credentials['chemalle']
# #api_key = credentials['3g2DEUppR01VxRz3P8NW']
# import plotly.plotly as py
# from datetime import datetime

# def graph_of_the_day(request):
#     df = pd.read_csv('docs/2018WinterOlympics.csv')

#     trace1 = go.Bar(
#         x=df['NOC'],  # NOC stands for National Olympic Committee
#         y=df['Gold'],
#         name = 'Gold',
#         marker=dict(color='#FFD700') # set the marker color to gold
#     )
#     trace2 = go.Bar(
#         x=df['NOC'],
#         y=df['Silver'],
#         name='Silver',
#         marker=dict(color='#9EA0A1') # set the marker color to silver
#     )
#     trace3 = go.Bar(
#         x=df['NOC'],
#         y=df['Bronze'],
#         name='Bronze',
#         marker=dict(color='#CD7F32') # set the marker color to bronze
#     )
#     data = [trace1, trace2, trace3]
#     layout = go.Layout(
#         title='2018 Winter Olympic Medals by Country',
#         barmode='stack'
#     )
#     fig = go.Figure(data=data, layout=layout)
#     py.iplot(fig, filename='gs-candlestick', validate=False)


#     return render(request, 'posts/GRAPH.html')




# file charts.py
def simple(request):
    import random
    import django
    import datetime

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig=Figure()
    ax=fig.add_subplot(111)
    x=[]
    y=[]
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    for i in range(10):
        x.append(now)
        now+=delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    canvas = fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    #response=django.http.HttpResponse(content_type='image/png')
    #canvas.print_png(response)
    figura = fig.savefig('graph.png')
    return render(request,'posts/GRAPH.html',{'figura':figura})



def Stocks_Data(request):
    if request.method == 'POST':
        form = stocksFORM(request.POST, request.FILES)
        if form.is_valid():
            # file is saved
            form.save()
            return render_to_response('posts/thankyou2.html')
            #return HttpResponseRedirect('home.html')
    else:
        form = stocksFORM()
    return render(request, 'posts/stocks.html', {'form': form})



def recommendation(request):
    try:
        qs = Stocks.pdobjects.order_by('-id')[:1].values('Ticker')
        df = qs.to_dataframe()
        df = df['Ticker'].tolist()
        # df = df['Asset'].str[0]
        df = df[0]
        Df = web.DataReader(df+'.sa', data_source='yahoo')
        Df=Df[['Open','High','Low','Close']]
        Df= Df.dropna()
        #############################################################################

        ###################### Creating input Parameters #########################
        Df['Std_U']=Df['High']-Df['Open']
        Df['Std_D']=Df['Open']-Df['Low']
        Df['S_3'] = Df['Close'].shift(1).rolling(window=3).mean()# pandas 0.19
        Df['S_15']= Df['Close'].shift(1).rolling(window=15).mean()# pandas 0.19
        Df['S_60']= Df['Close'].shift(1).rolling(window=60).mean()# pandas 0.19
        Df['OD']=Df['Open']-Df['Open'].shift(1)
        Df['OL']=Df['Open']-Df['Close'].shift(1)
        Df['Corr']=Df['Close'] .shift(1).rolling(window=10).corr(Df['S_3'] .shift(1))#pandas 0.19

        ####################### Creation of X and y datasets ######################
        X=Df[['Open','S_3','S_15','S_60','OD','OL','Corr']]# changed
        yU =Df['Std_U']
        yD =Df['Std_D']

        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        #############################################################################

        ########################## Centring and Scaling ###########################
        steps = [('imputation', imp),
                     ('scaler',StandardScaler()),
                     ('linear',LinearRegression())]
        #############################################################################

        ############################ Creating a Pipeline ##########################
        pipeline =Pipeline(steps)
        #############################################################################

        ############################# Hyper Parameters ##########################
        parameters = {'linear__fit_intercept':[0,1]}
        #############################################################################


        ############################# Cross Validation ############################
        reg = GridSearchCV(pipeline, parameters,cv=5)
        ############################################################################

        ############################ Test and Train Split ##########################
        t=.8
        split = int(t*len(Df))
        reg.fit(X[:split],yU[:split])
        #############################################################################
        ####################### Data Pre-Processing Completed ###################
        ############################# Regression #################################
        best_fit = reg.best_params_['linear__fit_intercept']
        reg = LinearRegression(fit_intercept =best_fit)
        X=imp.fit_transform(X,yU)
        reg.fit(X[:split],yU[:split])
        yU_predict =reg.predict(X[split:])

        reg = GridSearchCV(pipeline, parameters,cv=5)
        reg.fit(X[:split],yD[:split])
        best_fit = reg.best_params_['linear__fit_intercept']
        reg = LinearRegression(fit_intercept =best_fit)
        X=imp.fit_transform(X,yD)
        reg.fit(X[:split],yD[:split])
        yD_predict =reg.predict(X[split:])
        #############################################################################

        ############################ Prediction #####################################
        Df = Df.assign(Max_U =pd.Series(np.zeros(len(X))).values)
        Df = Df.assign(Max_D =pd.Series(np.zeros(len(X))).values)
        Df['Max_U'][split:]=yU_predict
        Df['Max_D'][split:]=yD_predict
        Df['Max_U'][Df['Max_U']<0]=0
        Df['Max_D'][Df['Max_D']<0]=0
        Df['P_H'] = Df['Open']+Df['Max_U']
        Df['P_L'] = Df['Open']-Df['Max_D']
        #########################################################################

        ######################## Strategy Implementation #######################
        Df = Df.assign(Ret =pd.Series(np.zeros(len(X))).values)

        Df['Ret']=np.log(Df['Close']/Df['Close'].shift(1))

        Df = Df.assign(Ret1 =pd.Series(np.zeros(len(X))).values)
        Df = Df.assign(Signal =pd.Series(np.zeros(len(X))).values)

        Df['Signal'][(Df['High']>Df['P_H']) &(Df['Low']>Df['P_L'])]=-1
        Df['Signal'][(Df['High']<Df['P_H']) &(Df['Low']<Df['P_L'])]=1

        Df['Ret1'][Df['Signal'].shift(1)==1]=Df['Ret']
        Df['Ret1'][Df['Signal'].shift(1)==-1]=-Df['Ret']

        Df = Df.assign(Cu_Ret1 =pd.Series(np.zeros(len(X))).values)
        Df['Cu_Ret1']=np.cumsum(Df['Ret1'][split:])

        Df = Df.assign(Cu_Ret =pd.Series(np.zeros(len(X))).values)
        Df['Cu_Ret']=np.cumsum(Df['Ret'][split:])

        Std =Df['Cu_Ret1'].expanding().std()
        Sharpe = (Df['Cu_Ret1']-Df['Cu_Ret'])/Std
        Sharpe=Sharpe.mean()

        high = Df['High'][-1]
        low = Df['Low'][-1]
        close = Df['Close'][-1]
        pivot = (high + low + close)/3

        R1 = (2 * pivot) - low
        S1 = (2 * pivot) - high
        R2 = (pivot - S1) + R1
        S2= pivot - (R1 - S1)

        Df['daily_ret'] = Df['Close'].pct_change()
        Df['excess_daily_ret'] = Df['daily_ret'] - 0.05/252


        def annualised_sharpe(returns, N=252):

            return np.sqrt(N) * returns.mean() / returns.std()

        def equity_sharpe(a):
            return annualised_sharpe(Df['excess_daily_ret'])

        answers = ['O mercado esta desafiador e o risco esta crescendo, veja se o Sharpe deste ativo é positivo, isto significa que a volatilidade do mesmo é muito boa a partir do indice 1 e representa menor risco no investimento',
                    'O volume negociado neste ativo cresceu muito nos ultimos 30 minutos do pregao, fique atento a reversao caso o ativo atinja o preço de resistencia rapidamente',
                    'Ha um crescente interesse neste ativo no momento, acompanhe os proximos 15 minutos para entrar com mais assertividade, o preço de resistencia indica um risco maior para compra, abaixo deste, o risco deve compensar',]

        answers2 = ['Ha uma tensao aparente e os vendidos comecam a ganhar a guerra dos comprados, nao compre este ativo se o valor de suporte for quebrado, a realizacao de lucros vai se intensificar',
                    'Ha um volume muito crescente de venda do ativo nos ultimos 30 minutos do pregao, fique atento se o Sharpe for inferior a 1, pois devera significar uma intensificacao da venda, em caso contrario compre apenas acima do preco de suporte e abaixo do preco de resistencia',
                    'O mercado esta desafiador, fique atento para comprar o ativo apenas caso o preco de suporte nao seja rompido',]

        answers3 = ['O ativo esta sendo negociado num patamar normal, nao ha indicador que direcione compra ou venda neste momento',
                    'Os indicadores de volume estao divergentes, nao ha uma visibilidade clara sobre posicionamento de curto prazo'
                    'Esteja atento a um aumento do volume para iniciar negociacao neste ativo, neste momento nao ha indicador convincente',]



        if Df['Signal'][-1]== -1:
            xls= ["{0:.2f}".format(R1),"{0:.2f}".format(S1),"{0:.2f}".format(equity_sharpe(df)),'Sell']
        elif Df['Signal'][-1]== 1:
            xls=["{0:.2f}".format(R1),"{0:.2f}".format(S1),"{0:.2f}".format(equity_sharpe(df)), 'Buy']
        else:
            xls=["{0:.2f}".format(R1),"{0:.2f}".format(S1),"{0:.2f}".format(equity_sharpe(df)), 'Neutral']


        if Df['Signal'][-1]== -1:
            talk = random.choice(answers)
        elif Df['Signal'][-1]== 1:
            talk = random.choice(answers2)
        else:
            talk = random.choice(answers3)


        data_barchart = pd.DataFrame(list(xls))
        data_barchart = data_barchart.T
        data_barchart.columns = ['Resistencia','Suporte','Sharpe','Action']
        data_barchart = data_barchart.to_html(index=False,columns=['Resistencia','Suporte','Sharpe','Action'])  # render with dynamic value



        qs2 = Stocks.pdobjects.order_by('-id')[:1].values('email')
        df2 = qs2.to_dataframe()
        df2 = df2['email'].tolist()
            # df = df['Asset'].str[0]
        df2 = df2[0]
        subject, from_email, to = 'Recomendação', 'econobilidade@econobilidade.com', str(df2)
        html_content = render_to_string('posts/name.html', {'data_barchart':data_barchart, 'talk':talk}) # render with dynamic value
        text_content = strip_tags(answers) # Strip the html tag. So people can see the pure text at least.

        # create the email, and attach the HTML version as well.
        msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
        msg.attach_alternative(html_content, "text/html")
        msg.send()


        return render_to_response('posts/thankyou2.html')

    except Exception:
        return render_to_response('posts/apologies.html')



def stock2(request):
    if request.method == 'POST':
        form = stocksFORM(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save()
            return UDACITY(form)

    else:
        form = stocksFORM()


    return render(request, 'posts/stocks.html', context = {'form': form})


import datetime

def UDACITY(form):
    try:
        ticker = form.Ticker+'.sa'
        EWZ = web.DataReader('EWZ', data_source='yahoo')[-252:]
        EWZ = EWZ['Close']
        VXX = web.DataReader('VXX', data_source='yahoo')[-252:]
        VXX = VXX['Close']
        GLD = web.DataReader('GLD', data_source='yahoo')[-252:]
        GLD = GLD['Close']
        BVSP = web.DataReader('^BVSP', data_source='yahoo')[-252:]
        BVSP = BVSP['Close']
        asset = web.DataReader(ticker, data_source='yahoo')[-252:]
        asset = asset['Close']
        # pivot cada ticker para uma coluna

        optimization = EWZ.to_frame()
        optimization.columns = ['EWZ']
        optimization['VXX'] = VXX
        optimization['GLD'] = GLD
        optimization['BVSP'] = BVSP
        optimization['asset'] = asset
        optimization.fillna(method='bfill', inplace=True)
        optimization.fillna(method='ffill', inplace=True)
            # pivot each ticker to a column

        X, y = [], []
        for index,row in optimization.iterrows():
            X.append(row[0:-1])
            y.append(row[-1])

        X, y = shuffle(X, y, random_state=23)

        num_training = int(0.9 * len(X))
        X_train, y_train = X[:num_training], y[:num_training]
        X_test, y_test = X[num_training:], y[num_training:]
        rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10,random_state=23)

        rf_regressor.fit(X_train, y_train)
        y_pred = rf_regressor.predict(X_test)

        adaboost_pred = rf_regressor.predict(optimization.iloc[:,0:-1])
        adaboost_pred = adaboost_pred.reshape(-1,1)

        optimization['Prediction'] = adaboost_pred

        mse = mean_squared_error(optimization['Prediction'], optimization.iloc[:,-2])
        evs = explained_variance_score(optimization['Prediction'], optimization.iloc[:,-2])

        optimization['MSE']= mse
        optimization['EVS'] = evs

        SPX = optimization

        SPX['SMA3'] = pd.Series.rolling(SPX['Prediction'], 3).mean()
        SPX['SMA8'] = pd.Series.rolling(SPX['Prediction'], 8).mean()
        SPX['SMA21'] = pd.Series.rolling(SPX['Prediction'], 21).mean()
        SPX['SMA50'] = pd.Series.rolling(SPX['Prediction'], 50).mean()
        SPX['SMA200'] = pd.Series.rolling(SPX['Prediction'], 200).mean()

        SPX.dropna(inplace=True)

    #     return SPX

        DayTrading = SPX['SMA3'] - SPX['SMA21']
        SPX['DayTrading'] = DayTrading
        SPX['SwingTrade'] = np.where(SPX['DayTrading']>0, 1,-1)
        SPX['GOLDEN_SMA3'] = SPX['SwingTrade'].shift(+1)
        SPX['position3'] = np.where(SPX['SMA3'] > SPX['SMA21'], 1, -1)
        SPX['position8'] = np.where(SPX['SMA8'] > SPX['SMA21'], 1, -1)
        Recomenda = SPX['SMA50'] - SPX['SMA200']
        SPX['Recomenda'] = Recomenda
        SPX['GOLDEN'] = np.where(SPX['Recomenda']>0, 1,-1)
        SPX['GOLDEN_CROSS'] = SPX['GOLDEN'].shift(+1)
        SPX['position'] = np.where(SPX['SMA3'] > SPX['SMA21'], 1, -1)
        SPX['market'] = np.log(SPX["asset"] / SPX["asset"].shift(1))
        # vectorized calculation of strategy returns
        SPX['strategy'] = SPX['position'].shift(1) * SPX['market']
        Golden_Rule = SPX['strategy'] - SPX['market']
        SPX['GOLDEN_RULE'] = Golden_Rule

        if SPX['GOLDEN_SMA3'][-1] == -1 and SPX['SwingTrade'][-1] == 1:
            average = [[ticker,SPX['SMA21'][-1], 'SMA3 just REACHED!', SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['position3'][-1] == 1:
            average = [[ticker, SPX['SMA21'][-1], 'SMA3',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['position8'][-1] == 1:
            average = [[ticker,SPX['SMA21'][-1], 'SMA8',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['GOLDEN_CROSS'][-1] == -1 and SPX['GOLDEN'][-1] == 1:
            average = [[ticker,SPX['SMA200'][-1], 'Golden_Cross REACHED!',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        elif SPX['GOLDEN'][-1] == 1:
            average = [[ticker,SPX['SMA200'][-1], 'Long Term',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]
        else:
            average = [[ticker,SPX['SMA200'][-1], 'downward trend',SPX['GOLDEN_RULE'][-1],optimization['MSE'][-1],optimization['EVS'][-1], 'Random Forest']]




        barchart = pd.DataFrame(list(average))
        barchart = barchart
        barchart.columns = ['Ticker', 'SMA','ACTION','Strategy Vs Market','MSE','EVS','Method']
        barchart2 = barchart.to_html(index=False,columns=['Ticker', 'SMA','ACTION','Strategy Vs Market','MSE','EVS','Method'])
        EVS =  "{0:.2f}%".format(float(barchart['EVS'][0]*100))
        MSE =  '{:,.2f}'.format(float(barchart['MSE'][0]))

    except Exception:
        return render_to_response('posts/apologies.html')
    return render_to_response('posts/recommendation.html', context= {'barchart':barchart2,'EVS':EVS, 'MSE': MSE})





import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# def ploty(request):
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(range(100))
#
#     fig.savefig("posts/static/images/graph.png")
#
#     return render (request, 'posts/GRAPH.html')




import quandl
quandl.ApiConfig.api_key = "oAe9Zos9MifP13eC9yRM"

import plotly.tools as tls
tls.set_credentials_file(username='chemalle', api_key='3g2DEUppR01VxRz3P8NW')

import plotly.plotly as py
from plotly.tools import FigureFactory as FF
from datetime import datetime


import pandas_datareader.data as web



def stock3(request):
    if request.method == 'POST':
        form = stocksFORM(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save()
            return ploty(form)

    else:
        form = stocksFORM()


    return render(request, 'posts/stocks.html', context = {'form': form})



def ploty(form):
    #try:

        ticker = form.Ticker+'.sa'
        df = web.DataReader(ticker, data_source='yahoo')[-21:]
        fig = FF.create_candlestick(df.Open, df.High, df.Low, df.Close, dates=df.index)
        py.plot(fig, filename='gs-candlestick', validate=False)

    #except Exception:
     #   return render_to_response('posts/apologies.html')

        return render_to_response('posts/GRAPH.html',context={'ticker':form.Ticker})



def Enterprise_Valuation(request):
    if request.method == 'POST':
        form = InputForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save()
            return Valuation(form)

    else:
        form = InputForm()


    return render(request, 'posts/enterprise.html', context = {'form': form})


def Valuation(form):

    sales = form.Net_Sales
    cogs = form.COGS
    expenses = form.Expenses
    pmr = form.PMR
    pmp = form.PMP

    df = [sales,cogs,expenses,pmr,pmp]
    df = pd.DataFrame(list(df))
    df = df.T
    df.columns = ['sales','cogs','expenses','pmr','pmp']

    ebitda_recebimento = (df['sales']/df['pmr'])*30
    ebitda_pagamento = (df['cogs']/df['pmp'])*30
    df['recebimento'] = ebitda_recebimento
    df['pagamento'] = ebitda_pagamento
    df['ebitda'] = df['recebimento'] - df['pagamento'] - df['expenses']
    valuation = df['ebitda']*10*12
    df['valuation'] = valuation
    valuation = '{:,.2f}'.format(df['valuation'].values[0])
    df = df.T
    df.columns = ['Valuation']
    df = df.to_html()
    df = df



    return render_to_response('posts/valuation.html',context={'valuation':valuation, 'company':form.Empresa, 'df':df})



def finance(request):
    return render (request, 'posts/udacity.html')



def finance2(request):
    return render (request, 'posts/udacity2.html')


def impairment(request):
    return render (request, 'posts/impairment.html')



def report(request):
    if request.method == 'POST':
        form = ReportForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save()
            return analise(form)

    else:
        form = ReportForm()


    return render(request, 'posts/analise.html', context = {'form': form})



def analise(form):
    try:

        name = form.Seu_nome
        mail = form.email
        crescer = form.Crescimento
        area = form.Segmento
        concorrente = form.Competitor


        if area == 'Serviço':
            EWZ = web.DataReader('b3sa3.sa', data_source='yahoo')[-21:]
        elif area == 'Indústria':
            EWZ = web.DataReader('wege3.sa', data_source='yahoo')[-21:]
        elif area == 'Agrobusiness':
            EWZ = web.DataReader('smto3.sa', data_source='yahoo')[-21:]
        elif area == 'Varejo':
            EWZ = web.DataReader('mglu3.sa', data_source='yahoo')[-21:]
        elif area == 'Financeiro':
            EWZ = web.DataReader('itub4.sa', data_source='yahoo')[-21:]
        elif area == 'Consultoria':
            EWZ = web.DataReader('qual.sa', data_source='yahoo')[-21:]
        elif area == 'Tecnologia':
            EWZ = web.DataReader('tots3.sa', data_source='yahoo')[-21:]
        else:
            EWZ = web.DataReader('aapl', data_source='yahoo')[-21:]


        def calc_daily_returns(closes):
            return np.log(closes/closes.shift(1))

        daily_returns = calc_daily_returns(EWZ['Close'])


        def calc_annual_returns(daily_returns):
            grouped = np.exp(daily_returns.groupby(
            lambda date: date.year).sum())-1
            return grouped


        annual_returns = calc_annual_returns(daily_returns)


        yeld = str('{:,.2f}'.format(annual_returns.sum()*100))



        answers_servicos = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu concorrente '+concorrente+' bem como seu mercado esta crescendo neste mês'+yeld+'%',
                    'Prezado(a) '+name+' o setor de serviço cresceu 2,2 pontos em abril e atingiu o melhor índice dos últimos três anos. Através de seus dados contábeis podemos direcionar e criar condições para seu crescimento ser de 5 a 30 pontos superior ao mercado aliando gastos, investimentos e performance tributária em outro patamar. Neste mês seu setor passa por um crescimento de '+yeld+' %']

        answers_industria = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu concorrente '+concorrente+' bem como seu mercado esta crescendo neste mês'+yeld+'% '+name+' podemos utilizar seus dados contabeis para direcionar e orientar seu crescimento como um bússola para seu desenvolvimento sustentável. Torne-se cliente , fique satisfeito ou devolvemos o seu dinheiro em dobro.'
                 'Caro visitante '+name+' a produção industrial brasileira fechou os dois primeiros meses do ano com crescimento acumulado de 4,3 pontos na comparação com o primeiro bimestre de 2017, a maior alta para um primeiro bimestre desde os 4,7 pontos de crescimento verificado em 2011. Neste mês seu setor passa por um crescimento de '+yeld+' %',]


        answers_consultoria = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu concorrente '+concorrente+' bem como seu mercado esta crescendo neste mês'+yeld+'%',]


        answers_varejo = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu concorrente '+concorrente+' bem como seu mercado esta crescendo neste mês'+yeld+'%',]


        answers_agrobusiness = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu concorrente '+concorrente+' bem como seu mercado esta crescendo neste mês'+yeld+'%',
                    'Carissimo visitante '+name+' os dados avaliados até fevereiro de 2018 do PIB do agronegócio brasileiro indicam queda de 0,12 pontos na renda do setor no mês e, com isso, baixa de 0,23 pontos no acumulado do primeiro bimestre e projeção baixista na evolução anual (1,37%).'
                    'Ressalta-se que tais dados refletem as projeções iniciais de produção das atividades do agronegócio e preços relativos ao primeiro bimestre de 2018 com relação ao mesmo período do ano anterior. Dados importantes, como os de produção pecuária, ainda não estavam disponíveis para avaliação até o fechamento deste relatório.'
                    'Portanto, as projeções agregadas devem passar por alterações significativas nos próximos. Seu concorrente '+concorrente+' bem como seu mercado esta crescendo '+yeld+'%',]

        answers_financeiro = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu concorrente '+concorrente+' bem como seu mercado esta crescendo neste mês'+yeld+'%',]

        answers_tecnologia = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu concorrente '+concorrente+' bem como seu mercado esta crescendo neste mês'+yeld+'%',]




        if area == 'Serviço':
            talk = random.choice(answers_servicos)
        elif area == 'Indústria':
            talk = random.choice(answers_industria)
        elif area == 'Agrobusiness':
            talk = random.choice(answers_agrobusiness)
        elif area == 'Varejo':
            talk = random.choice(answers_varejo)
        elif area == 'Financeiro':
            talk = random.choice(answers_financeiro)
        elif area == 'Consultoria':
            talk = random.choice(answers_industria)
        elif area == 'Tecnologia':
            talk = random.choice(answers_tecnologia)
        else:
            talk = random.choice(answers3)



        subject, from_email, to = 'Relatorio', 'econobilidade@econobilidade.com', mail
        html_content = render_to_string('posts/review.html', {'talk':talk}) # render with dynamic value
        text_content = strip_tags(talk) # Strip the html tag. So people can see the pure text at least.

        # create the email, and attach the HTML version as well.
        msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
        msg.attach_alternative(html_content, "text/html")
        msg.attach_file('docs/relatorio_Focus_analise.pdf')
        msg.send()


        return render_to_response('posts/thanks.html')

    except Exception:
        return render_to_response('posts/apologies2.html')



def candle(request):
    if request.method == 'POST':
        form = CandleForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save()
            return candle_to(form)

    else:
        form = CandleForm()


    return render(request, 'posts/candle.html', context = {'form': form})


def candle_to(form):
        mail = form.email
        talk = 'Obrigado por solicitar mais uma obra #econobilidade'
        subject, from_email, to = '#econobilidade-Serie SABEDORIA', 'econobilidade@econobilidade.com', mail
        html_content = render_to_string('posts/candlesticks.html') # render with dynamic value
        text_content = strip_tags(talk) # Strip the html tag. So people can see the pure text at least.

        # create the email, and attach the HTML version as well.
        msg = EmailMultiAlternatives(subject, text_content, from_email, [to])
        msg.attach_alternative(html_content, "text/html")
        #msg.attach_file('docs/relatorio_Focus_analise.pdf')
        msg.send()


        return render_to_response('posts/thanks.html')



def cash(request):
    if request.method == 'POST':
        form = NewsletterForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save()
            return render (request, 'posts/cash_flow.html', context = {'form': form})

    else:
        form = NewsletterForm()



    return render (request, 'posts/cash_flow.html', context = {'form': form})



def subscription(request):
    if request.method == 'POST':
        form = NewsletterForm(request.POST or None, request.FILES or None)
        if form.is_valid():
            form = form.save()

            return itau_report(form)

    else:
        form = NewsletterForm()


    return render(request, 'posts/subscription.html', context = {'form': form})





def itau_report(form):
    try:
        name = form.Seu_nome
        mail = form.email
        area = form.Setor


        if area == 'Serviço':
            EWZ = web.DataReader('b3sa3.sa', data_source='yahoo')[-21:]
        elif area == 'Indústria':
            EWZ = web.DataReader('wege3.sa', data_source='yahoo')[-21:]
        elif area == 'Agrobusiness':
            EWZ = web.DataReader('smto3.sa', data_source='yahoo')[-21:]
        elif area == 'Varejo':
            EWZ = web.DataReader('mglu3.sa', data_source='yahoo')[-21:]
        elif area == 'Financeiro':
            EWZ = web.DataReader('itub4.sa', data_source='yahoo')[-21:]
        elif area == 'Consultoria':
            EWZ = web.DataReader('qual.sa', data_source='yahoo')[-21:]
        elif area == 'Tecnologia':
            EWZ = web.DataReader('tots3.sa', data_source='yahoo')[-21:]
        else:
            EWZ = web.DataReader('aapl', data_source='yahoo')[-21:]


        def calc_daily_returns(closes):
            return np.log(closes/closes.shift(1))

        daily_returns = calc_daily_returns(EWZ['Close'])


        def calc_annual_returns(daily_returns):
            grouped = np.exp(daily_returns.groupby(
            lambda date: date.year).sum())-1
            return grouped


        annual_returns = calc_annual_returns(daily_returns)


        yeld = str('{:,.2f}'.format(annual_returns.sum()*100))
        yeld2 = annual_returns.sum()



        answers_servicos = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu mercado esta crescendo neste mês'+yeld+'%',
                    'Prezado(a) '+name+' o setor de serviço cresceu 2,2 pontos em abril e atingiu o melhor índice dos últimos três anos. Através de seus dados contábeis podemos direcionar e criar condições para seu crescimento ser de 5 a 30 pontos superior ao mercado aliando gastos, investimentos e performance tributária em outro patamar. Neste mês seu setor passa por um crescimento de '+yeld+' %']

        answers_industria = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu mercado esta crescendo neste mês'+yeld+'% '+name+' podemos utilizar seus dados contabeis para direcionar e orientar seu crescimento como um bússola para seu desenvolvimento sustentável. Torne-se cliente , fique satisfeito ou devolvemos o seu dinheiro em dobro.'
                 'Caro visitante '+name+' a produção industrial brasileira fechou os dois primeiros meses do ano com crescimento acumulado de 4,3 pontos na comparação com o primeiro bimestre de 2017, a maior alta para um primeiro bimestre desde os 4,7 pontos de crescimento verificado em 2011. Neste mês seu setor passa por um crescimento de '+yeld+' %',]


        answers_consultoria = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu mercado esta crescendo neste mês'+yeld+'%',]


        answers_varejo = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu mercado esta crescendo neste mês'+yeld+'%',]


        answers_agrobusiness = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu mercado esta crescendo neste mês'+yeld+'%',
                    'Carissimo visitante '+name+' os dados avaliados até fevereiro de 2018 do PIB do agronegócio brasileiro indicam queda de 0,12 pontos na renda do setor no mês e, com isso, baixa de 0,23 pontos no acumulado do primeiro bimestre e projeção baixista na evolução anual (1,37%).'
                    'Ressalta-se que tais dados refletem as projeções iniciais de produção das atividades do agronegócio e preços relativos ao primeiro bimestre de 2018 com relação ao mesmo período do ano anterior. Dados importantes, como os de produção pecuária, ainda não estavam disponíveis para avaliação até o fechamento deste relatório.'
                    'Portanto, as projeções agregadas devem passar por alterações significativas nos próximos. Seu mercado esta crescendo '+yeld+'%',]

        answers_financeiro = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu mercado esta crescendo neste mês'+yeld+'%',]

        answers_tecnologia = ['Carissimo visitante '+name+' o risco esta crescendo no seu segmento, a greve dos caminhoneiros tirou da economia algo equivalente ao PIB de 20 cidades médias brasileiras de um trimestre, seu concorrente e o seu mercado como um todo devem fechar o mês com crescimento de '+yeld+'%',
                    'Carissimo visitante '+name+' seu mercado esta crescendo neste mês'+yeld+'%',]




        if area == 'Serviço':
            talk = random.choice(answers_servicos)
        elif area == 'Indústria':
            talk = random.choice(answers_industria)
        elif area == 'Agrobusiness':
            talk = random.choice(answers_agrobusiness)
        elif area == 'Varejo':
            talk = random.choice(answers_varejo)
        elif area == 'Financeiro':
            talk = random.choice(answers_financeiro)
        elif area == 'Consultoria':
            talk = random.choice(answers_industria)
        elif area == 'Tecnologia':
            talk = random.choice(answers_tecnologia)
        else:
            talk = random.choice(answers3)



        itau = pd.read_excel(form.itau)
        itau.columns = ['nada','data','nada2','nada3','historico','nada4','valor','saldo']
        itau_analise =itau[['data','historico','valor', 'saldo']]
        itau_analise = itau_analise.fillna(0)
        itau_analise.valor = pd.to_numeric(itau_analise.valor, errors='coerce')
        itau_analise['Debito'] = np.where(itau_analise['valor']>0,itau_analise['valor'],0)
        itau_analise['Credito'] = np.where(itau_analise['valor']<0,-itau_analise['valor'],0)
        itau_analise['PN'] = np.where(itau_analise['data']=='Data',1,0)
        itau_analise['balance'] = itau_analise['Debito'] - itau_analise['Credito']
        filter_col = [col for col in itau_analise.historico.unique()]
        names = []

        for i in filter_col:
            i = itau_analise.loc[itau_analise['historico'].values==i]
            i = i['Debito'] - i['Credito']
            names.append(i.sum())


            dictionary = dict(zip(filter_col, names))

        dictionary = dict(zip(filter_col, names))

        itau = pd.DataFrame.from_dict(dictionary,orient='index')

        itau.columns = ['balance']

        itau = itau[itau['balance'] !=0]

        itau['entradas'] = np.where(itau['balance']>0,itau['balance'],0)
        itau['saidas'] = np.where(itau['balance']<0,itau['balance'],0)

        collections = itau['entradas'] > 0
        payments = itau['saidas'] < 0

        receivables = itau['entradas'][collections].sum()
        payables = itau['saidas'][payments].sum()

        resultado = str('{:,.2f}'.format((itau['entradas'].sum() + itau['saidas'].sum()) + (itau['entradas'].sum() + itau['saidas'].sum())*yeld2)) #+ itau['entradas'].sum() + itau['saidas'].sum()))
        #resultado = str('{:,.2f}'.format(resultado))

        return render_to_response ('posts/resultado.html', context = {'resultado': resultado, 'receivables':receivables,'payables':payables, 'talk':talk})

    except Exception:
        return render_to_response('posts/apologies2.html')
