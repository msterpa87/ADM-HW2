import pandas as pd
import numpy as np
import seaborn as sns

# casting columns types to reduce DataFrame size
dtypes = {'event_type':'category',
        'user_id':int,
        'category_id':'category',
        'product_id':'category',
        'price':float}

filename = 'octnov.csv'
october_filename = '2019-Oct.csv'
november_filename = '2019-Nov.csv'

chunksize=1000000

# RQ1-1: plot the average number of operations within a session
def avg_user_operation():
    cols = ['event_type','user_id']
    df = pd.read_csv(filename,
                     usecols=cols)

    count = df.groupby(['user_id','event_type']).size()
    count = count.unstack(fill_value=0).stack().to_frame('count').reset_index()
    avg = count[['event_type','count']].groupby('event_type').mean()

    title = "Average number of operation per session"
    xlabel = "Event type"
    ylabel = "Number of operations"
    plt = avg.plot.bar(title=title)
    plt.set_xlabel(xlabel)
    plt.set_ylabel(ylabel)


# RQ1-2 returns average number of times a user views a product before adding it to the cart
def average_views_before_cart():
    cols=['user_id','product_id','event_type']

    df = pd.read_csv(filename,
                     dtype=dtypes,
                     usecols=cols,
                     nrows=50000000)

    df = df[df.event_type!='purchase']
    df['row'] = np.arange(len(df))
    
    cart_df = df[df.event_type=='cart'].groupby(['user_id','product_id'], as_index=False).agg({'row':'min'})
    cart_df.set_index(['product_id','user_id'])
    
    view_df = df[df.event_type=='view'].drop('event_type',axis=1)
    view_df.set_index(['product_id','user_id'])
    
    merged = view_df.merge(cart_df, on=['product_id','user_id'], suffixes=['','_cart'])
    count = merged[merged.row < merged.row_cart].groupby(['product_id','user_id']).size().to_frame()
    return count.groupby('user_id').mean().mean()


# RQ1-3 returns the probability of purchase of a product put in the cart
def purchase_probabily(chunksize=chunksize):
    cols = ['user_id','product_id','event_type']

    chunks = pd.read_csv(filename,
                         dtype=dtypes,
                         usecols=cols,
                         chunksize=chunksize)

    processed = []

    # filter only rows with cart or purchase event
    for df in chunks:
        cart_or_purchase_mask = (df.event_type=='cart') | (df.event_type=='purchase')
        df = df[cart_or_purchase_mask]
        processed.append(df)

    # count number of products a user puts in the cart and then purchase
    cols = ['user_id','product_id']
    grouped = df.groupby(cols)

    n_purchase = len(grouped.filter(lambda x: (x.event_type=='cart').any() & (x.event_type=='purchase').any()))
    n_cart = len(grouped.filter(lambda x: (x.event_type=='cart').any()))

    return n_purchase/n_cart


# RQ1-4: average time between first and cart or purchase
def view_time_diff(x):
    # x is a groupby object with a view,purchase,cart row
    x = x.event_time.tolist()
    return x[1]-x[0]


def avgtime_after_view():
    # return a string representing the average time
    cols = ['user_session','event_type','event_time']

    df = pd.read_csv(filename,
                     usecols=cols,
                     dtype=dtypes)

    df.event_time = pd.to_datetime(df.event_time, infer_datetime_format=True)
    cart_sessions = df[df.event_type=='cart'].user_session
    filtered = df[df.user_session.isin(cart_sessions.tolist())]
    final = filtered.groupby(['user_session','event_type']).min()
    avgseconds = final.reset_index().groupby(['user_session']).apply(view_time_diff).mean().seconds
    minutes, seconds = divmod(avgseconds, 60)
    return "{} min {} sec".format(minutes,seconds)

# RQ2: most visited subcategories
def most_visited_subcategories(filename):
    cols = ['category_id','product_id','event_type']

    df = pd.read_csv(filename,
                     usecols=cols,
                     dtype=dtypes)

    n_cat = 10
    purchase = df[df.event_type=='purchase'][['category_id','product_id']]
    grouped = purchase.groupby(['category_id']).count().product_id
    top_purchase = grouped.sort_values(ascending=False)[:n_cat]

    # count products purchase within category
    products = purchase.groupby(['category_id','product_id']).size()

    # list top 10 purchased products per category
    products = products.groupby('category_id', group_keys=False).apply(lambda x: x.sort_values(ascending=False).head(10))
    top10 = pd.DataFrame(products, columns=['Number of purchases'])

    return top_purchase, top10


# RQ3
def avgPrice(code):
    
    dtypes={'brand' : 'category',
            'category_code': 'category',
            'price':float}

    columns=['brand',"category_code",'price']

    october_chunks = pd.read_csv(filename,
                     dtype=dtypes,
                     usecols=columns,
                     chunksize=chunksize)
    
    processed_list = []

    for chunk in october_chunks:
        chunk_code = chunk.category_code==code
        chunk = chunk[chunk_code]
        processed_list.append(chunk)
        
    
    df = pd.concat(processed_list)
    
    s=df.groupby(['brand']).price.mean().sort_values()
    
    plt=s.plot(kind='bar',title='the average price of the {} sold by the brands in Oct & Nov'.format(code))
    plt.set_ylabel('average price')
    
    return

def listofhigh_average():

    #returninig the brands which have the highest price in each category
    dtypes={'brand' : 'category',
            'category_code': 'category',
            'price':float}

    columns=['brand',"category_code",'price','user_session','user_id']

    october_chunks = pd.read_csv(filename,
                     dtype=dtypes,
                     usecols=columns,
                     chunksize=chunksize)

    processed_list = []

    for chunk in october_chunks:
        chunk = chunk.groupby(["category_code",'brand']).price.mean().reset_index()
        processed_list.append(chunk)

    df = pd.concat(processed_list).set_index(["category_code",'brand'])

    brand_sorted = df.groupby(['category_code','brand']).price.max().sort_values().to_frame('price')
    
    return brand_sorted

# RQ4

def Totall_Brand_Sales(brand_name):
    
    dtypes={'event_type':'category',
            'brand' : 'category',
            'price':float}

    columns=['brand','price','event_type']

    october_chunks = pd.read_csv(october_filename,
                     dtype=dtypes,
                     usecols=columns,
                     chunksize=chunksize)
    

    november_chunks = pd.read_csv(november_filename,
                     dtype=dtypes,
                     usecols=columns,
                     chunksize=chunksize)

    processed_list1 = []
    processed_list2 = []
    
    for chunk in october_chunks:
        chunk_brand = (chunk.brand==brand_name) | (chunk.event_type=='purchase')
        chunk = chunk[chunk_brand]
        processed_list1.append(chunk)
    df1 = pd.concat(processed_list1)
    
    for chunk in november_chunks:
        chunk_brand = (chunk.brand==brand_name) | (chunk.event_type=='purchase')
        chunk = chunk[chunk_brand]
        processed_list2.append(chunk)
    df2 = pd.concat(processed_list2)

    print("Total sale of ",brand_name,":")
    print("October: %.2f" % df1.price.sum())
    print("November: %.2f" % df2.price.sum(),"\n")

def top3lost():

    dtypes={'event_type':'category',
        'brand' : 'category',
        'price':float}

    columns=['brand','price','event_type']

    october_chunks = pd.read_csv(october_filename,
                     dtype=dtypes,
                     usecols=columns,
                     chunksize=chunksize)

    november_chunks = pd.read_csv(november_filename,
                     dtype=dtypes,
                     usecols=columns,
                     chunksize=chunksize)
    processed_list1 = []
    processed_list2 = []
    
    for chunk in october_chunks:
        chunk_brand = (chunk.event_type=='purchase')
        chunk = chunk[chunk_brand]
        processed_list1.append(chunk)
        
    df1 = pd.concat(processed_list1)
    
    for chunk in november_chunks:
        chunk_brand = (chunk.event_type=='purchase')
        chunk = chunk[chunk_brand]
        processed_list2.append(chunk)

    df2 = pd.concat(processed_list2)

    income_Oct=df1.groupby(['brand']).price.sum()
    income_Nov=df2.groupby(['brand']).price.sum()

    lost=(income_Nov-income_Oct)/income_Oct
    lost=lost.sort_values()
    lost.index[0:3]

    for i in range(3):
        bn=lost.index[i]
        percentage=abs(lost[i])*100 
        print("brand '{}' lost {:.2f} % between October and November".format(bn,percentage))
    return


# RQ5

import seaborn as sns
import matplotlib.pyplot as plt

codes= {
    1: "night 12 a.m.-6 a.m.",
    2: "morning 6 a.m.-12 p.m.",
    3: "day 12 p.m.-6 p.m",
    4: "evening 6 p.m.-12 a.m",
}


class RQ5:
    
    def __init__(self, files=None):
        if files is None:
            files = ['../data_in/2019-Oct.csv', '../data_in/2019-Nov.csv']
        self.files = files
        self.df = None
    
    def __read_data(self, sample=1.0):
        columns = ['event_type', 'user_id', 'event_time']
        dtypes = {'event_type': 'category', 'user_id': int}
        self.df = (pd.concat([pd.read_csv(f,
                                    usecols=columns, 
                                    dtype=dtypes, 
                                    parse_dates=['event_time']) for f in self.files])
                   .sample(frac=sample)
                  )
        self.df = self.df[self.df['event_type'] == 'view']
        self.df['date'] = self.df['event_time'].dt.date
        self.df['period'] = (self.df['event_time'].dt.hour % 24+6) // 6
        
    def visitors_by_part_of_day(self):
        if self.df is None:
            self.__read_data()
        grp = (self.df.groupby(['date', 'period'])['user_id'].nunique()
               .reset_index()
               .rename(columns={'user_id': 'visitors'}))
        grp2 = grp.groupby('period')['visitors'].mean().round().astype(int).reset_index()
        grp2['day_part'] = grp2['period'].apply(lambda x: codes[x])
        print("Subtask-1. Visitors by part of the day")
        print(grp2.head(len(grp2)))
        
    def plot_visitors_by_hours(self, grp, figsize=(18,6)):
        ax = plt.figure(figsize=figsize)
        right_end = len(grp)+1
        xticks = range(0,right_end,6)

        sns.lineplot(grp.index, grp['visitors']);
    
        plt.xticks(xticks, grp.timeslot[0:right_end:6], rotation=45)
        plt.grid(b=True, which='major', color='grey', linestyle='--')
        plt.title('Mean visitors by hour of weekdays')
        plt.ylabel('Mean number of visitors')
        plt.xlabel('Weekday, hour')
        print("Subtask-2. Visitors by hour of weekdays")
        plt.show()
    
    
    def plot_average_visitors_by_hours(self):
        if self.df is None:
            self.__read_data()
        self.df['hour'] = self.df['event_time'].dt.hour
        self.df['weekday'] = self.df['event_time'].dt.weekday
        self.df['weekday_name'] = self.df['event_time'].dt.strftime('%a')
        self.df['date'] = self.df['event_time'].dt.date
        
        # groupby and prepare data for visualisation
        grp3 = (self.df
                .groupby(['date', 'weekday', 'hour', 'weekday_name'])
                .agg({'user_id': 'nunique'})
                .reset_index()
                .rename(columns={'user_id': 'visitors'}))

        grp4 = grp3.groupby(['weekday', 'hour', 'weekday_name'])['visitors'].mean().round().astype(int).reset_index()
        grp4['timeslot'] = grp4.apply(lambda x: f"{x['weekday_name']}, {x['hour']:0>2}:00", axis=1)
        self.plot_visitors_by_hours(grp4)
        

# RQ6

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

costs = {
    'view': 0,
    'purchase': 1
}
        


class RQ6:
    
    def __init__(self, files=None):
        if files is None:
            files = ['../data_in/2019-Oct.csv', '../data_in/2019-Nov.csv']
        self.files = files
        self.df = None
        self.report = None
        
    def __read_data(self):
        columns = ['event_type', 'user_session', 'user_id', 'category_id', 'product_id']
        dtypes = {
                  'user_session':'category',
                  'user_id':int,
                  'category_id':'category',
                  'product_id':'category'                 
                 }
        self.df = pd.concat([pd.read_csv(f,
                                    usecols=columns, 
                                    dtype=dtypes) for f in self.files]).sample(frac=0.1)
        
        self.df = self.df[self.df['event_type'].isin(['view', 'purchase'])]
        self.df['event_type'] = self.df['event_type'].apply(lambda x: costs[x])
        
        
    def store_conversion_report(self):
        if self.df is None:
            self.__read_data()
        
        grp = self.df.groupby('product_id')
        conversion = grp['event_type'].sum() / grp['event_type'].count()
        mean_conversion = conversion.mean() * 100
        
        # calculating conversion rate by user 
        grp = self.df.groupby(['product_id'])['event_type'].max()
        grpx = grp.reset_index().groupby('product_id')
        product_conversion_by_user = grpx['event_type'].sum() / grpx['event_type'].count()
        product_conversion_by_user_overall = (product_conversion_by_user * 100).mean()
        
        print(f"Overall store conversion: {mean_conversion:.2f}%")
        print(f"Overall store conversion by user: {product_conversion_by_user_overall:.2f}%")
        
    def __prepare_report(self):
        if self.df is None:
            self.__read_data()
        
        # prepare data by category
        self.df['category_code'] = self.df['category_code'].fillna(self.df['category_id'])

        grp_cat = self.df.groupby(['category_code'])
        num_of_purchase_by_cat = grp_cat['event_type'].sum()
        num_of_events = grp_cat['event_type'].apply(lambda x: (x==0).sum())
        conv_rate_by_cat = num_of_purchase_by_cat / num_of_events * 100

        # collecting in one report dataframe
        rq6_df = num_of_purchase_by_cat.reset_index().rename(columns={'event_type': 'purchase_num'})
        rq6_df['view_num'] = num_of_events.reset_index()['event_type']
        rq6_df['conv_rate'] = conv_rate_by_cat.reset_index()['event_type']
        self.report = rq6_df
        
    def plot_categories_by_conversion_rate(self, top_n=100, figsize=(20,7)):
        if self.report is None:
            self.__prepare_report()
            
        # plotting categories by conversion rate (descending order)
        self.report = self.report.sort_values(by='conv_rate', ascending=False)
        vis = self.report.iloc[:top_n]
        ax = vis.plot.bar(x='category_code', y='conv_rate', figsize=figsize)
        plt.show();
    
    def plot_categories_by_num_of_purchases(self, top_n=100, figsize=(20,7)):
        if self.report is None:
            self.__prepare_report()
        
        # plotting categories by number of purchases (descending order)
        self.report = self.report.sort_values(by='purchase_num', ascending=False)
        vis = self.report.iloc[:top_n]
        ax = vis.plot.bar(x='category_code', y='purchase_num', figsize=(20,7));
        plt.show();


# RQ7: 
def pareto_principle():
    cols = ['user_id','price']

    df = pd.read_csv(filename,
                     usecols=cols,
                     dtype=dtypes)

    total_income = df.price.sum()
    grouped = df.groupby('user_id').sum()
    grouped = grouped.sort_values('price', ascending=False)
    n_users = int(len(grouped)*0.2)
    top20_income = float(grouped[:n_users].sum())
    return top20_income/total_income