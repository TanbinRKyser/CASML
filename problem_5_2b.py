import pybloom_live as bf
import pandas as pd
import os
#import multiprocessing
#from memory_profiler import profile

class BookLookup:
    def __init__(self, book_df):
        self.lookup_C = self.build_campus_lookup(book_df, "C")
        self.lookup_SD = self.build_campus_lookup(book_df, "SD")
        self.lookup_SFB = self.build_campus_lookup(book_df, "SFB")
        
        self.lookup_central = self.build_central_lookup(self.lookup_C, self.lookup_SD, self.lookup_SFB)

    def build_campus_lookup(self, dataframe, campus):
        """Creates an id lookup structure for a specified campus.
	
        Args:
            book_df (pandas DataFrame): data set with all book ids
            campus (str): BTU campus, [C, SD, SFB]

        Returns:
            campus_lookup: lookup structure per campus books
        """
        ### TODO: Add/Change code below
        
        #campus_lookup = bf.BloomFilter( capacity = 100_000_00, error_rate = 0.001) 
        campus_lookup = bf.BloomFilter( capacity = 50_000_000, error_rate = 0.0001)  
        
        filtered_rows = dataframe[dataframe['campus'] == campus] 
        
        for book_id in filtered_rows['book_id']:  
            # print(f"[DEBUG] adding {book_id} into campus {campus}")
            campus_lookup.add( book_id )  

        ### TODO: Add/Change code above
        
        return campus_lookup

    def build_central_lookup(self, lookup_C, lookup_SD, lookup_SFB):
        """Creates a central id lookup structure.

        Args:
            lookup_C, lookup_SD, lookup_SFB:  lookup structure per campus

        Returns:
            central_lookup: central lookup structure
        """
        ### TODO: Add/Change code below

        central_lookup = {
            "C": lookup_C,
            "SD": lookup_SD,
            "SFB": lookup_SFB
        }  

        ### TODO: Add/Change code above
        
        return central_lookup

    def search_book(self, id_to_search):
        """Looks up a book id and returns the respective campus

        Args:
            id_to_search (str): string id to search in the lookup structure

        Returns:
            campus (str): Campus at which the book is present
        """
        ### TODO: Add/Change code below
        
        if id_to_search in self.lookup_central[ "C" ]:
            campus = "C"
        elif id_to_search in self.lookup_central[ "SD" ]:
            campus = "SD"
        elif id_to_search in self.lookup_central[ "SFB" ]:
            campus = "SFB"
        else:
            campus = None

        ### TODO: Add/Change code above

        if campus:
            return campus
        else:
            return "Book Not Found in Lookup!"

def lookup_books(books, ids_to_search):
    """Creates a book_lookup instance and Looks up a list of book ids, returns the respective campus

    Args:
        books (iterator): data set with all book ids
        id_to_search (list): string id to search in the lookup structure

    Returns:
        lookups (list): a list of campuses in which each book is present, ### NOTE: Order of the campus items should be the same as the ids_to_search list.
    """ 
    
    book_lookup = BookLookup(books)

    ### TODO: Add/Change code below
    lookups = [ book_lookup.search_book( book_id ) for book_id in ids_to_search ]
    
    ### TODO: Add/Change code above
    
    return lookups

# def process_chunk( chunk, ids_to_search ):
#     book_lookup = BookLookup( chunk )
#     return [ book_lookup.search_book( book_id ) for book_id in ids_to_search ]

# @profiler
# def main():

if __name__ == "__main__":
    ### NOTE: The main clause will not be graded, change for your own convenience  
    ### TODO: Add/Change code below

    # main()
    book_file = os.path.abspath("books.csv")  # Generated by identifier_generator.py
    
    #chunksize = 1000000
    chunksize = 50_000_000  
    
    book_chunks = pd.read_csv( book_file, chunksize = chunksize, header = None, names=[ "book_id", "campus" ] )
   
    #book_id = ["B123432"]
    #id_to_search = ["B20000243"]
    ids_to_search = ["B00000001", "B12345678", "B00000003","B99999999","B00000004","B12343212"] 
     

    # with multiprocessing.Pool(processes=4) as pool:
    #     results = pool.starmap( process_chunk, [( chunk, ids_to_search ) for chunk in book_chunks ] )

    final_results = {}

    for chunk in book_chunks:
        results = lookup_books( chunk, ids_to_search )
        
        for book_id, campus in zip( ids_to_search, results ):

            if book_id not in final_results or final_results[ book_id ] == "Book Not Found in Lookup!":
                final_results[ book_id ] = campus

    for book_id, result in final_results.items():
        print(f"Book ID: { book_id }, Campus: { result }")

    # iter = 0
    # max_iter = 10
    # final_results = {}
    # for chunk in book_chunks:
    #     iter += 1
    #     if iter < max_iter:
    #         results = lookup_books( chunk, ids_to_search )
            
    #         for book_id, campus in zip( ids_to_search, results ):
                
    #             if book_id not in final_results: 
    #                 print(f"[DEBUG] adding { book_id }: { campus }")
    #                 final_results[ book_id ] = campus
    #             elif campus != "Book Not Found in Lookup!" and final_results[ book_id ] == "Book Not Found in Lookup!":      
    #                 print(f"[DEBUG] updating { book_id } from { final_results[ book_id ] } to { campus }")
    #                 final_results[book_id] = campus
    #             else:
    #                 print(f"[DEBUG] skipping { book_id }")
    
    # for book_id, result in final_results.items():
    #     print(f"bookid: { book_id }, campus: { result }")

    ## >>>mprof run python test.py
    ## >>>mprof plot