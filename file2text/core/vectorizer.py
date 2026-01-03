"""Модуль для векторизации текста."""

import numpy as np
from typing import List, Tuple, Optional, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class Vectorizer:
    """Класс для векторизации текста и работы с векторными представлениями."""
    
    def __init__(
        self,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Инициализация векторизатора.
        
        Args:
            model: Модель для векторизации (sentence-transformers)
            device: Устройство для обработки. Если None, определяется автоматически
            verbose: Выводить ли подробную информацию
        """
        self.model_name = model
        self.verbose = verbose
        
        if self.verbose:
            print(f"Загрузка модели векторизации: {model}...")
        
        self.model = SentenceTransformer(model, device=device)
        self.vector_dimension = self.model.get_sentence_embedding_dimension()
        
        if self.verbose:
            print(f"Модель векторизации загружена. Размерность векторов: {self.vector_dimension}")
    
    def vectorize(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Векторизует текст.
        
        Args:
            text: Текст или список текстов для векторизации
            
        Returns:
            np.ndarray: Вектор(ы) текста
        """
        if isinstance(text, str):
            return self.model.encode(text, convert_to_numpy=True)
        else:
            return self.model.encode(text, convert_to_numpy=True)
    
    def vectorize_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Векторизует список текстов пакетно.
        
        Args:
            texts: Список текстов
            batch_size: Размер батча для обработки
            
        Returns:
            np.ndarray: Массив векторов
        """
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Вычисляет схожесть между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            float: Косинусная схожесть (0-1)
        """
        vec1 = self.vectorize(text1)
        vec2 = self.vectorize(text2)
        return float(cosine_similarity([vec1], [vec2])[0][0])
    
    def search(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Ищет наиболее похожие тексты на запрос.
        
        Args:
            query: Поисковый запрос
            texts: Список текстов для поиска
            top_k: Количество результатов
            threshold: Минимальный порог схожести
            
        Returns:
            List[Tuple[str, float]]: Список (текст, схожесть) отсортированный по убыванию
        """
        query_vec = self.vectorize(query)
        text_vecs = self.vectorize_batch(texts)
        
        similarities = cosine_similarity([query_vec], text_vecs)[0]
        
        # Фильтруем по порогу и сортируем
        results = [(texts[i], float(similarities[i])) 
                   for i in range(len(texts)) 
                   if similarities[i] >= threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def save_vectors(self, vectors: np.ndarray, path: str):
        """
        Сохраняет векторы в файл.
        
        Args:
            vectors: Массив векторов
            path: Путь для сохранения
        """
        np.save(path, vectors)
    
    def load_vectors(self, path: str) -> np.ndarray:
        """
        Загружает векторы из файла.
        
        Args:
            path: Путь к файлу
            
        Returns:
            np.ndarray: Массив векторов
        """
        return np.load(path)
    
    # Методы для будущей интеграции с векторными БД
    def prepare_for_vector_db(self, vectors: np.ndarray) -> dict:
        """
        Подготавливает векторы для сохранения в векторную БД.
        
        Args:
            vectors: Массив векторов
            
        Returns:
            dict: Данные в формате для векторной БД
        """
        return {
            "vectors": vectors.tolist(),
            "dimension": self.vector_dimension,
            "model": self.model_name
        }
