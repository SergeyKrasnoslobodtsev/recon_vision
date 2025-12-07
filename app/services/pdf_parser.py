import base64
from loguru import logger


class PDFParserService:
    """Сервис для парсинга PDF документов"""

    async def parse_pdf(self, pdf_base64: str) -> dict:
        """
        Парсит PDF и возвращает структурированные данные

        Args:
            pdf_base64: PDF документ в формате base64

        Returns:
            dict: Структурированные данные акта сверки
        """
        logger.info("Начало парсинга PDF документа")

        # Декодируем base64 в байты (для будущей библиотеки)
        pdf_bytes = base64.b64decode(pdf_base64)

        # TODO: Здесь будет вызов библиотеки парсинга
        # parsed_data = external_parser_lib.parse(pdf_bytes)

        # Заглушка: возвращаем тестовые данные
        mock_data = {
            "seller": "ООО Рога и Копыта",
            "buyer": "ПАО Тестовая Компания",
            "period": {"start": "2025-01-01", "end": "2025-01-31"},
            "entries": [
                {
                    "document_number": "ДОК-001",
                    "document_date": "2025-01-15",
                    "debit": 10000.50,
                    "credit": 0.0,
                    "description": "Оплата по договору",
                },
                {
                    "document_number": "ДОК-002",
                    "document_date": "2025-01-20",
                    "debit": 0.0,
                    "credit": 5000.25,
                    "description": "Возврат товара",
                },
            ],
            "total_debit": 10000.50,
            "total_credit": 5000.25,
            "balance": 5000.25,
        }

        logger.info(
            f"Парсинг завершён. Найдено {len(mock_data.get('entries', []))} записей"
        )
        return mock_data

    def validate_pdf(self, pdf_base64: str) -> bool:
        """
        Проверяет корректность PDF документа

        Args:
            pdf_base64: PDF документ в формате base64

        Returns:
            bool: True если документ валиден
        """
        try:
            pdf_bytes = base64.b64decode(pdf_base64)

            if not pdf_bytes.startswith(b"%PDF"):
                logger.error("Документ не является PDF файлом")
                return False
            return True
        except Exception as e:
            logger.error(f"Ошибка валидации PDF: {e}")
            return False
