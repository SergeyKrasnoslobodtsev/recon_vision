from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_fill_reconciliation_act_use_case,
    get_process_status_use_case,
    get_submit_reconciliation_act_use_case,
)
from app.application.dto.fill_reconciliation_act import FillReconciliationActResult
from app.application.dto.get_process_status import GetProcessStatusResult
from app.application.dto.submit_reconciliation_act import SubmitReconciliationActResult
from app.application.errors import (
    ProcessFailedError,
    ProcessNotFoundError,
    ProcessNotReadyError,
)
from app.domain.entities.ledger_entry import LedgerEntry, RowReference
from app.domain.entities.process import ProcessState
from app.domain.entities.reconciliation_data import ReconciliationData
from app.domain.enums.process_status import ProcessStatus
from app.domain.value_objects.period import Period
from app.main import get_application


@pytest.fixture
def application():
    app = get_application()
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(application):
    with TestClient(application) as test_client:
        yield test_client


def _override_use_case(application, dependency, execute: AsyncMock) -> AsyncMock:
    use_case = SimpleNamespace(execute=execute)
    application.dependency_overrides[dependency] = lambda: use_case
    return execute


class TestSendReconciliationActContract:
    def test_returns_process_id_on_success(self, application, client):
        execute = _override_use_case(
            application,
            get_submit_reconciliation_act_use_case,
            AsyncMock(return_value=SubmitReconciliationActResult(process_id="process-123")),
        )

        response = client.post(
            "/api/v1/send_reconciliation_act",
            json={"document": "cGRm"},
        )

        assert response.status_code == 201
        assert response.json() == {"process_id": "process-123"}
        command = execute.await_args.args[0]
        assert command.document_base64 == "cGRm"

    def test_returns_bad_request_for_invalid_document(self, application, client):
        _override_use_case(
            application,
            get_submit_reconciliation_act_use_case,
            AsyncMock(side_effect=ValueError("Не удалось декодировать PDF из base64")),
        )

        response = client.post(
            "/api/v1/send_reconciliation_act",
            json={"document": "!!!"},
        )

        assert response.status_code == 400
        assert response.json() == {
            "status": -2,
            "message": "Не удалось декодировать PDF из base64",
        }


class TestProcessStatusContract:
    def test_returns_wait_while_processing(self, application, client):
        process_state = ProcessState(
            process_id="process-123",
            status=ProcessStatus.PROCESSING,
        )
        _override_use_case(
            application,
            get_process_status_use_case,
            AsyncMock(return_value=GetProcessStatusResult(process_state=process_state)),
        )

        response = client.post(
            "/api/v1/process_status",
            json={"process_id": "process-123"},
        )

        assert response.status_code == 201
        assert response.json() == {"status": 0, "message": "wait"}

    def test_returns_not_found_for_missing_process(self, application, client):
        _override_use_case(
            application,
            get_process_status_use_case,
            AsyncMock(side_effect=ProcessNotFoundError("Процесс не найден")),
        )

        response = client.post(
            "/api/v1/process_status",
            json={"process_id": "missing"},
        )

        assert response.status_code == 404
        assert response.json() == {"status": -1, "message": "not found"}

    def test_returns_processing_error(self, application, client):
        process_state = ProcessState(
            process_id="process-123",
            status=ProcessStatus.FAILED,
            message="semantic extraction failed",
        )
        _override_use_case(
            application,
            get_process_status_use_case,
            AsyncMock(return_value=GetProcessStatusResult(process_state=process_state)),
        )

        response = client.post(
            "/api/v1/process_status",
            json={"process_id": "process-123"},
        )

        assert response.status_code == 500
        assert response.json() == {
            "status": -2,
            "message": "semantic extraction failed",
        }

    def test_returns_done_payload_without_process_id(self, application, client):
        process_state = ProcessState(
            process_id="process-123",
            status=ProcessStatus.COMPLETED,
            message="Документ успешно обработан",
            reconciliation_data=ReconciliationData(
                seller="АО Продавец",
                buyer="ООО Покупатель",
                period=Period(start="2025-01-01", end="2025-01-31"),
                debit=[
                    LedgerEntry(
                        record="Реализация",
                        value=1200.5,
                        date="2025-01-15",
                        row_reference=RowReference(id_table="table-1", id_row="row-2"),
                    )
                ],
                credit=[
                    LedgerEntry(
                        record="Оплата",
                        value=700.0,
                        date="2025-01-20",
                        row_reference=RowReference(id_table="table-2", id_row="row-4"),
                    )
                ],
            ),
        )
        _override_use_case(
            application,
            get_process_status_use_case,
            AsyncMock(return_value=GetProcessStatusResult(process_state=process_state)),
        )

        response = client.post(
            "/api/v1/process_status",
            json={"process_id": "process-123"},
        )

        assert response.status_code == 200
        assert response.json() == {
            "process_id": "process-123",
            "status": 0,
            "message": "Документ успешно обработан",
            "seller": "АО Продавец",
            "buyer": "ООО Покупатель",
            "period": {"start": "2025-01-01", "end": "2025-01-31"},
            "debit": [
                {
                    "row_id": {"id_row": "row-2", "id_table": "table-1"},
                    "record": "Реализация",
                    "value": 1200.5,
                    "date": "2025-01-15",
                }
            ],
            "credit": [
                {
                    "row_id": {"id_row": "row-4", "id_table": "table-2"},
                    "record": "Оплата",
                    "value": 700.0,
                    "date": "2025-01-20",
                }
            ],
        }


class TestFillReconciliationActContract:
    def test_returns_document_on_success(self, application, client):
        execute = _override_use_case(
            application,
            get_fill_reconciliation_act_use_case,
            AsyncMock(return_value=FillReconciliationActResult(document_base64="ZmlsbGVkLXBkZg==")),
        )

        response = client.post(
            "/api/v1/fill_reconciliation_act",
            json={
                "process_id": "process-123",
                "comments": "Комментарий к акту",
                "debit": [
                    {
                        "row_id": {"id_row": "row-2", "id_table": "table-1"},
                        "record": "Реализация",
                        "value": 1200.5,
                        "date": "2025-01-15",
                    }
                ],
                "credit": [],
            },
        )

        assert response.status_code == 200
        assert response.json() == {"document": "ZmlsbGVkLXBkZg=="}
        command = execute.await_args.args[0]
        assert command.process_id == "process-123"
        assert command.comments == "Комментарий к акту"
        assert len(command.debit) == 1
        assert command.debit[0].row_reference == RowReference(id_table="table-1", id_row="row-2")
        assert command.debit[0].record == "Реализация"
        assert command.debit[0].value == 1200.5
        assert command.debit[0].date == "2025-01-15"

    def test_returns_wait_when_fill_is_not_ready(self, application, client):
        _override_use_case(
            application,
            get_fill_reconciliation_act_use_case,
            AsyncMock(side_effect=ProcessNotReadyError("wait")),
        )

        response = client.post(
            "/api/v1/fill_reconciliation_act",
            json={"process_id": "process-123", "debit": [], "credit": []},
        )

        assert response.status_code == 201
        assert response.json() == {"status": 0, "message": "wait"}

    def test_returns_not_found_when_fill_process_is_missing(self, application, client):
        _override_use_case(
            application,
            get_fill_reconciliation_act_use_case,
            AsyncMock(side_effect=ProcessNotFoundError("missing")),
        )

        response = client.post(
            "/api/v1/fill_reconciliation_act",
            json={"process_id": "missing", "debit": [], "credit": []},
        )

        assert response.status_code == 404
        assert response.json() == {"status": -1, "message": "not found"}

    def test_returns_error_when_fill_fails(self, application, client):
        _override_use_case(
            application,
            get_fill_reconciliation_act_use_case,
            AsyncMock(side_effect=ProcessFailedError("fill failed")),
        )

        response = client.post(
            "/api/v1/fill_reconciliation_act",
            json={"process_id": "process-123", "debit": [], "credit": []},
        )

        assert response.status_code == 500
        assert response.json() == {"status": -2, "message": "fill failed"}