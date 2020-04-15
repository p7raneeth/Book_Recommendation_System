import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np


def read_data():
    ratings = pd.read_csv(r'Data/BX-Book-Ratings.csv', encoding='latin-1')
    books = pd.read_csv(r'Data/BX-Books.csv', error_bad_lines=False, encoding='latin-1')
    users = pd.read_csv(r'Data/BX-Users.csv', sep=';',encoding='latin-1')
    books.drop(['Unnamed: 8', 'Unnamed: 9',
        'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13',
        'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17',
        'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21','Book-Author',
       'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M',
       'Image-URL-L'], axis=1, inplace=True)
    # users.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], inplace=True, axis=1)
    return(ratings, books, users)


def plot_data(ratings, users):
    ratings['Book-Rating'].value_counts(sort=False).plot(kind='bar')
    plt.title('Book_Ratings vs Count\n')
    plt.xlabel('Book_Ratings')
    plt.ylabel('Count')
    # plt.show()

    users['Age'].hist(bins=[0,10,20,30,40,50,60,70,80,90,100])
    plt.title('Age_Distribution\n')
    plt.xlabel('Age')
    plt.ylabel('Count')
    # plt.show()


def filter_ratings(ratings, value):
    counts1 = ratings['User-ID'].value_counts()
    users_with_ratings_above_200 = ratings[ratings['User-ID'].isin(counts1[counts1 >= value].index)]
    return users_with_ratings_above_200


def fetch_books_with_high_ratings(book_ratings):
    book_ratings['Book-Title'].value_counts()
    book_ratings.dropna(axis=0, subset=['Book-Title'], inplace=True)
    book_ratings_count = book_ratings.groupby(['Book-Title'])['Book-Rating'].count().reset_index()
    ratings_with_totalRatingsCount = book_ratings.merge(book_ratings_count, left_on='Book-Title', 
                    right_on='Book-Title', how='left')
    popular_books = ratings_with_totalRatingsCount[ratings_with_totalRatingsCount['Book-Rating_y'] >= 50]
    popular_books.drop_duplicates(['User-ID', 'Book-Title'], inplace=True)
    return popular_books


def create_data_pivot(popular_books):
    popular_books_pivot = popular_books.pivot(index='Book-Title', columns='User-ID', values='Book-Rating_x').fillna(0)
    popular_books_matrix = csr_matrix(popular_books_pivot.values)
    return popular_books_matrix, popular_books_pivot


def classification_model(popular_books_matrix):
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(popular_books_matrix)
    return model_knn

def recommend_books(model_knn, popular_books_pivot):
    random_movie_index = np.random.randint(popular_books_pivot.shape[0])
    print('random book name', popular_books_pivot.index[random_movie_index])
    distances, indices = model_knn.kneighbors(popular_books_pivot.iloc[random_movie_index,:].values.reshape(1,-1),n_neighbors=5)
    print(distances, indices)
    suggested_books = []
    for i in range(0, len(distances.flatten())):
        suggested_books.append(popular_books_pivot.index[indices.flatten()[i]])
    return suggested_books


if __name__ == "__main__":
    ratings, books, users = read_data()
    plot_data(ratings, users)
    users_with_ratings_above_200 = filter_ratings(ratings, value=200)
    book_ratings = pd.merge(users_with_ratings_above_200, books, on='ISBN')
    popular_books = fetch_books_with_high_ratings(book_ratings)
    popular_books_matrix, popular_books_pivot = create_data_pivot(popular_books)
    model_knn = classification_model(popular_books_matrix)
    suggested_books = recommend_books(model_knn, popular_books_pivot)
    print('suggested books', suggested_books)
    
