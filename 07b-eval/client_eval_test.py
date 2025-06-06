#
# Copyright Elasticsearch B.V. and contributors
# SPDX-License-Identifier: Apache-2.0
#
import asyncio
import os

from delayed_assert.delayed_assert import assert_all, expect
import pytest
from client import OpenAIClient
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from main import message

eval_model = os.getenv("EVAL_MODEL", "o4-mini")

async def evaluate_metrics(metrics, test_case, actual_output):
    tasks = [metric.a_measure(test_case, False) for metric in metrics]
    await asyncio.gather(*tasks)
    failures = [
        f"{type(metric).__name__} scored {metric.score:.1f}: {actual_output}"
        for metric in metrics
        if not metric.success
    ]
    return failures


@pytest.mark.eval
@pytest.mark.asyncio
async def test_chat_eval(traced_test):
    test_case = Case(
        name="check_ocean",
        inputs=message,
        expected_output="Atlantic Ocean",
        metadata={'difficulty': 'easy'},
    )

    evaluators = [
        LLMJudge(
            rubric="Is the answer relevant to the question?",
            model=eval_model,
            include_input=True,
        ),
        LLMJudge(
            rubric="Is the answer a hallucination?",
            model=eval_model,
            include_input=True,
        ),
    ]

    dataset = Dataset(
        cases=[test_case],
        evaluators=evaluators,
    )

    report = await dataset.evaluate(lambda msg: OpenAIClient().chat(msg))

    with assert_all():
        expect(report.cases[0].assertions["LLMJudge"].value, report.cases[0].assertions["LLMJudge"].reason)
        expect(not report.cases[0].assertions["LLMJudge_2"].value, report.cases[0].assertions["LLMJudge_2"].reason)
