from loguru import logger


class PDFFillerService:
    """Сервис для заполнения PDF документов"""

    async def fill_pdf(self, original_pdf_bytes: bytes, data: dict) -> bytes:
        """
        Заполняет оригинальный PDF документ данными

        Args:
            original_pdf_bytes: Оригинальный PDF в байтах
            data: Данные для заполнения (из ProcessData.structure)

        Returns:
            bytes: Заполненный PDF документ
        """
        logger.info("Начало заполнения PDF документа")

        # TODO: Здесь будет вызов библиотеки заполнения PDF
        # filled_pdf = external_filler_lib.fill(original_pdf_bytes, data)

        # Заглушка: возвращаем оригинальный PDF
        logger.warning(
            "Используется заглушка! Возвращается оригинальный PDF без изменений"
        )
        filled_pdf_bytes = original_pdf_bytes

        logger.info("Заполнение PDF завершено")
        return filled_pdf_bytes
