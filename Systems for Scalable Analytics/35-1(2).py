import json
import ctypes
from dask.distributed import Client

def trim_memory() -> int:
    libc = ctypes.CDLL("libc.so.6")
    return libc.malloc_trim(0)

def PA0(path_to_user_reviews_csv):
    client = Client()
    # Helps fix any memory leaks.
    client.run(trim_memory)
    client = client.restart()
    
    #my code starts here
    
    # path = 'user_reviews.csv'
    cols = ['reviewerID', 'asin', 'helpful', 'overall','reviewTime']

    start = time.time()

    #helper function to extract element at idx from split
    def helper(ser, idx):
    return (ser.strip('[]').split(', ')[idx])

    ddf = dd.read_csv(path_to_user_reviews_csv, usecols = cols)

    #creating new columns for helpful/total votes
    ddf['helpful_votes'] = ddf.helpful.apply(helper,args=(0,), meta=ddf.helpful).astype(int)
    ddf['total_votes'] = ddf.helpful.apply(helper,args=(1,), meta=ddf.helpful).astype(int)

    #getting year from reviewTime
    #ddf['reviewing_since'] = ddf.reviewTime.apply(helper,args=(1,), meta=ddf.reviewTime).astype(int)
    ddf['reviewing_since'] = ddf.reviewTime.apply(lambda x: x.split(', ')[1], meta=ddf.reviewTime).astype(int)

    #using agg to parallel compute funcs for each category
    tempdf = ddf.groupby('reviewerID').agg({'asin': 'count', 'overall':'mean',
                                    'reviewing_since':'min', 'helpful_votes':'sum', 'total_votes':'sum' }) 

    #formatting the column names
    tempdf = tempdf.reset_index().rename(columns = {'asin':'number_products_rated',
                                            'overall':'avg_ratings'})
    submit = tempdf.describe().compute().round(2)

    #end = time.time()
    #print(end - start)
    #submit
    # use this one  

    with open('results_PA0.json', 'w') as outfile: 
        json.dump(json.loads(submit.to_json()), outfile)