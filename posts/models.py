from django.db import models

from django.utils import timezone
from django.urls import reverse
from django.db import models
from django.forms import ModelForm
from django import forms
from django_pandas.managers import DataFrameManager
from django.utils.formats import localize

# Create your models here.

DICAS = (
    ('S', 'SIM'),
    ('N','NÃO'),
)


GROWTH = (
    ('S', 'SIM'),
    ('N','NÃO'),
)


SEGMENTO = (
    ('Serviço', 'Serviço'),
    ('Indústria','Indústria'),
    ('Agrobusiness','Agrobusiness'),
    ('Varejo','Varejo'),
    ('Financeiro','Financeiro'),
    ('Consultoria','Consultoria'),
    ('Tecnologia','Tecnologia'),


)

class Post(models.Model):
    title = models.CharField(max_length=250)
    pub_date = models.DateTimeField()
    image = models.ImageField(upload_to='media/')
    body = models.TextField()


    def __str__(self):
        return self.title



    def pub_date_pretty(self):
        return self.pub_date.strftime('%b %e %Y')


    def summary(self):
        return self.body[:50]


class Stocks(models.Model):
    Ticker = models.CharField(max_length=10,help_text='Qual o ativo voce quer analisar? Exemplo: petr4 para analisar Petrobras',blank=False)
    email = models.EmailField(max_length=200,help_text='Qual o seu e-mail?',blank=False)
    Seu_nome = models.CharField(max_length=200,help_text='Qual o seu nome?',blank=False)
    Dicas =  models.CharField(max_length=12,choices=DICAS, help_text='Gostaria de receber dicas de Investimento?')

    pdobjects = DataFrameManager()


    def __str__(self):
        return self.Ticker



class Input(models.Model):
    Empresa = models.CharField(max_length=50, help_text='Informe o nome fantasia da sua empresa')
    email = models.EmailField(max_length=200,help_text='Qual o seu e-mail?',blank=False)
    Seu_nome = models.CharField(max_length=200,help_text='Qual o seu nome?',blank=False)
    Net_Sales = models.DecimalField(default=0.00, max_digits=10000, decimal_places=2,blank=True, help_text='Vendas Liquidas de Impostos | utilizar formato americano')
    COGS = models.DecimalField(default=0.00, max_digits=10000, decimal_places=2,blank=True, help_text='Custos dos produtos vendidos | utilizar formato americano')
    Expenses = models.DecimalField(default=0.00, max_digits=10000, decimal_places=2,blank=True, help_text='Despesas Operacionais | utilizar formato americano')
    PMR = models.DecimalField(default=0.00, max_digits=10000, decimal_places=2,blank=True, help_text='Insira o Prazo Medio de Recebimentos | utilizar formato americano')
    PMP = models.DecimalField(default=0.00, max_digits=10000, decimal_places=2,blank=True, help_text='Insira o Prazo Medio de Pagamentos | utilizar formato americano')
    Dicas =  models.CharField(max_length=12,choices=DICAS, help_text='Gostaria de receber dicas de Investimento?')


    pdobjects = DataFrameManager()


    def __str__(self):
        return self.Empresa




class Report(models.Model):
    email = models.EmailField(max_length=200,help_text='Qual o seu e-mail?',blank=False)
    Seu_nome = models.CharField(max_length=200,help_text='Qual o seu nome?',blank=False)
    Crescimento =  models.CharField(max_length=12,choices=GROWTH, help_text='Sua empresa esta crescendo neste ano?')
    Segmento = models.CharField(max_length=12,choices=SEGMENTO, help_text='Sua empresa esta crescendo neste ano?')
    Competitor =  models.CharField(max_length=12, help_text='Nome de um concorrente direto ou produto substituto')


    pdobjects = DataFrameManager()


    def __str__(self):
        return self.email



class Candle(models.Model):
    email = models.EmailField(max_length=200,help_text='Qual o seu e-mail?',blank=False)
    pdobjects = DataFrameManager()


    def __str__(self):
        return self.email



class newsletter(models.Model):
    email = models.EmailField(max_length=200,help_text='Qual o seu e-mail?',blank=False)
    Seu_nome = models.CharField(max_length=200,help_text='Qual o seu nome?',blank=False)
    itau = models.FileField(blank=True, null=True,help_text='Faça o upload do seu extrato',
    upload_to='documents/')
    Setor = models.CharField(max_length=12,choices=SEGMENTO, help_text='Qual o seu segmento?',blank=True, null=True)


    pdobjects = DataFrameManager()


    def __str__(self):
        return self.email
