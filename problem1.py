from pyspark import SparkContext
# math library for sqrt
import math

def summaryStatistics(file, num_cores):
    bins = 10
    ### TODO: Add/Change code below
    ### NOTE: Use the variable sc to create the SparkContext
    # sc = None
    # sc = SparkContext("SummaryStatistics",master=f"local[{num_cores}]")
    sc = SparkContext.getOrCreate()

    rdd = sc.textFile( file )\
        .map( lambda l: l.split('\t') )\
            .map( lambda x: float( x[ 2 ] ) )

    count = rdd.count()
    total_sum = rdd.reduce( lambda a,b: a + b )
    mean = total_sum / count

    ####### Standard deviation
    #### sum of squared (x - mean) values
    var = rdd.map( lambda x : ( x - mean)**2 ).reduce( lambda a,b: a + b ) / count
    std = math.sqrt( var )


    total_max = rdd.reduce( lambda a,b : a if a > b else b )
    total_min = rdd.reduce( lambda a,b : a if a < b else b )

    ##### HISTOGRAM ####
    # bins = 10
    bin_size = ( total_max - total_min ) / bins

    def assign_val_to_bins( value ):
        index = min( int( ( value - total_min ) / bin_size ), bins - 1 )
        return ( index, 1 )

    histogram = rdd.map( assign_val_to_bins ).reduceByKey( lambda a, b: a + b ).collect()
    histogram = sorted( histogram, key = lambda x: x[0] )
    
    
    ### MEDIAN ###
    sorted_rdd = rdd.sortBy( lambda x: x )
    m_index = count//2

    if count % 2 == 0 :
        medians = sorted_rdd.zipWithIndex().filter( lambda x : x[1] in (m_index-1, m_index)).map(lambda x: x[0]).collect()
        median = sum( medians ) / 2

    else:
        median = sorted_rdd.zipWithIndex().filter( lambda x : x[1] == m_index ).map( lambda x : x[0] ).collect()[0]
    ### TODO: Add/Change code above
    ### NOTE: It's best practice to specifically close the SparkContext
    sc.stop()
    return mean, std, total_max, total_min, histogram, median
    # return mean, std, total_max, total_min, median


if __name__ == "__main__":
    
    ### NOTE: The main clause will not be graded, change for your own convenience , don't delete the main clause completely
    ### TODO: Add/Change code below
    workers = 4
    file = "data-assignment-8-1M.dat"
    stats = summaryStatistics(file, workers)
    print(stats)
