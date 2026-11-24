"""The questions the model being trained answers about a feedback event, each
with a fixed JSON answer (ported unchanged from scripts/signals/model.py, where
they were checked against a hand-labelled answer key):

- classify: split a user message into feedback / new task / question parts
- review: which of a turn's unflagged tool results carried a signal
- responsible: which steps caused the feedback or should have used it
- lesson: what to do differently, how general that is, how success is checked
"""

TYPES = ["correction", "retry", "approval"]

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "says_something_is_wrong": {"type": "boolean"},
                    "kind": {"type": "string", "enum": ["feedback", "new_task", "question"]},
                    "type": {"type": ["string", "null"], "enum": TYPES + [None]},
                    "text": {"type": "string"},
                },
                "required": ["says_something_is_wrong", "kind", "type", "text"],
            },
        }
    },
    "required": ["parts"],
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "signals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "type": {"type": "string", "enum": ["outcome", "failed_result", "success"]},
                    "quote": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["step", "type", "quote", "why"],
            },
        }
    },
    "required": ["signals"],
}

RESPONSIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["steps", "reason"],
}

PREFERENCE_SCHEMA = {
    "type": "object",
    "properties": {"kind": {"type": "string", "enum": ["about_me", "reaction", "task", "text"]}},
    "required": ["kind"],
}

LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "lesson": {"type": "string"},
        "short": {"type": "string"},
        "instance": {"type": "string"},
        "generality": {"type": "string", "enum": ["task", "kind", "always"]},
        "verifiable": {"type": ["string", "null"]},
    },
    "required": ["lesson", "short", "instance", "generality", "verifiable"],
}

CLASSIFY = """You are reviewing a conversation between a user and a coding agent.

Earlier user messages (oldest first):
{earlier}

What the agent did since the last user message (one line per step):
{steps}

New user message:
{message}

Split the new user message into parts. For each part give:
- says_something_is_wrong: true if the part says anything is wrong, broken, missing, too long, or should be done differently; false otherwise
- kind: "feedback" if it judges the agent's work above, "new_task" if it asks for something new, "question" if it only asks something
- type, only for feedback:
  "correction" = says something is wrong, broken, missing, or should be different (even briefly, even politely)
  "retry" = only asks to try again, without saying what is wrong
  "approval" = accepts the work as it is
  null for new_task and question
- text: the exact words of that part
Most messages have one part.

Examples:
"cli is broken; keep the old entrypoint working too" -> [{{"says_something_is_wrong": true, "kind": "feedback", "type": "correction", "text": "cli is broken; keep the old entrypoint working too"}}]
"Wrong commit message. Use Conventional Commits. Try again." -> [{{"says_something_is_wrong": true, "kind": "feedback", "type": "correction", "text": "Wrong commit message. Use Conventional Commits. Try again."}}]
"Too long. Keep it under 30 lines." -> [{{"says_something_is_wrong": true, "kind": "feedback", "type": "correction", "text": "Too long. Keep it under 30 lines."}}]
"Try again." -> [{{"says_something_is_wrong": false, "kind": "feedback", "type": "retry", "text": "Try again."}}]
"Looks good to me." -> [{{"says_something_is_wrong": false, "kind": "feedback", "type": "approval", "text": "Looks good to me."}}]
"Thanks, the commit looks fine. Now add a --dry-run flag to dedupe.py." -> [{{"says_something_is_wrong": false, "kind": "feedback", "type": "approval", "text": "Thanks, the commit looks fine."}}, {{"says_something_is_wrong": false, "kind": "new_task", "type": null, "text": "Now add a --dry-run flag to dedupe.py."}}]
"What does --in-place do?" -> [{{"says_something_is_wrong": false, "kind": "question", "type": null, "text": "What does --in-place do?"}}]

Answer with JSON only."""

REVIEW = """You are reviewing what a coding agent did for this request:
{task}

{ran}
{steps}

Find outputs that carry information the agent must use, or that show its work is wrong or right, even though no error was reported. Most turns have none: return an empty list then.
Types:
- "outcome": information the next steps should use. Example: "APPLE  Y...G" from a word game, "500: higher" from a guessing game.
- "failed_result": output that shows the work is wrong without an error message. Example: after implementing a converter, "III -> 0, IV -> 0, IX -> 0".
- "success": output that proves the requested work now does what the user asked. Example: the user wanted `python3 -m pkg circle 2` to print 12.57 and the output is "circle 2: 12.57".
Not signals: output that merely shows a command ran, file contents, git status, listings, compile checks like "syntax OK", or {errors}
For each signal give the step number, the type, a short quote of the output, and why it matters. Answer with JSON only."""

# The two wordings of the review. A turn of shell commands is asked as it
# always was: the rules find a command's errors (exit codes, tracebacks), so
# the review leaves them alone. Nothing else reads the result of an agent's
# own function or MCP tool, so there a refusal or an error is itself a signal.
REVIEW_COMMANDS = {
    "ran": "These are the commands the agent ran in one turn, with their output:",
    "errors": "any error message (those are handled elsewhere).",
}
REVIEW_TOOLS = {
    "ran": "These are the tool calls the agent made in one turn, with their results:",
    "errors": (
        "the error message of a shell command (those are handled elsewhere). When one of the agent's other "
        "tools refuses a call or says the result is wrong (a move that is not allowed, a value it does not "
        'accept, an attempt that failed), that is a signal of type "failed_result".'
    ),
}

RESPONSIBLE = """You are reviewing what a coding agent did for this request:
{task}

The agent's steps:
{steps}

Feedback that arrived {where}:
{feedback}

What the agent did after this feedback (it may show what really had to be fixed):
{after}

Which steps are responsible for this feedback? {allowed}
- A step is responsible if its action was wrong or incomplete and caused the problem, or if it should have used earlier information and did not (for example it repeated a guess that had already been scored, or said "done" without checking what the user asked for).
- A step that only read, searched, or ran a check that revealed the problem is not responsible.
- For a tool error, the responsible step is usually the step that made the failing call.
- For information such as a game's feedback: if the later steps used it correctly, return an empty list.

Examples:
- After renaming a function the user says "the CLI is broken", and afterwards the agent fixes a file it had not edited. Responsible: the step that made the incomplete rename, and the step that said it was done without running the CLI.
- Step 8 guessed APPLE and got "Y...G"; steps 9 and 10 guessed APPLE again. Feedback "Y...G" at step 8. Responsible: steps 9 and 10, which ignored it.
- "Could not find oldString" in the result of step 3. Responsible: step 3.

Give the step numbers and one sentence on why. Answer with JSON only."""

LESSON = """A coding agent got this feedback while working on:
{task}

What the responsible steps did:
{steps}

Feedback:
{feedback}

Why these steps are responsible: {reason}

What the agent did after the feedback:
{after}

Write what the agent should do differently next time.{wish}
- lesson: one or two sentences of advice that would also help on a different task of the same kind. Name the general practice; do not mention this task's files, function names or values.
  Good: "When renaming a function, search for every way it is used, including names built as strings, and run the program's entry points before saying it is done."
  Bad: "Update report.py to use circle_area."
- short: the same advice as one instruction of at most 15 words, starting with a verb, keeping what matters most (it is shown to the agent at the start of every task).
  Good: "Search for every use of a renamed name, including strings, and run the entry points."
- instance: the specific facts of this case (files, values, what exactly went wrong).
- generality: "task" if the advice only helps with this exact task; "kind" if it helps with tasks of the same kind (renames, word games, writing tests); "always" if it applies to any task: how to use the tools correctly (which tool runs shell commands, exact paths, edit strings, not thinking at length before acting), or a standing preference of this user (for example a commit message style).
  Examples: "The execute tool runs JavaScript only; use the shell tool for shell commands" -> always. "Write commit messages in Conventional Commits style" -> always. "In Wordle, keep letters marked G in place" -> kind. "When renaming a function, search for names built as strings" -> kind.
- verifiable: if success on such a task can be checked automatically, how ("the tests pass", "the game is won"); otherwise null.
Answer with JSON only."""



# Added to the lesson question when the feedback is the user's own words. A
# wish said in passing ("uv please", "in metric please") is a standing
# preference, and the lesson must keep the thing the user named. Live
# (Qwen3.5-4B, 24 chats corrected in everyday words): "I use uv, not pip." was
# stored as "Verify the user's preferred package manager before providing
# instructions." and no answer changed. Shapes with placeholders, not worked
# examples: a worked example is copied (see EXAMPLE_LESSONS).
USER_WISH = """
The feedback is the user's own words. If it tells the agent how this user wants things (which tool, which format, which wording, which units, which style), that wish itself is the lesson, and the rule below about naming the general practice does not apply:
  - lesson and short are the direct instruction and name exactly what the user named, in the user's words. The user's key words (the name of the tool, the unit, the phrase, the format) must appear in both, written out. Never replace them with their category ("the preferred tool", "the requested format") and never turn the wish into advice to check, verify or ask.
    Shapes, where <THING> and <OTHER> stand for the user's own words:
    "<THING>, not <OTHER>." -> "Use <THING>, not <OTHER>."
    "<THING> please." -> "Use <THING>." or "Write <THING>."
    "You forgot <THING>." -> "Always include <THING>."
  - generality is "always" for such a wish, unless the user limited it to this one task."""

# Asked about a sentence the rules flagged as a possible preference of the
# user's own (rules.preference_sentences). Only "about_me" is kept. The other
# three are what such a sentence turned out to be in real chats: a pick among
# offers or a word on the last answer ("I love the new layout"), a line about
# the job ("I only get three results"), a line of a text handed over to be
# worked on (a cover letter or an essay in the first person).
PREFERENCE = """A user is talking with an AI assistant.

What the assistant said just before (nothing at the start of a conversation):
{before}

The user's message:
{message}

One sentence of that message:
{sentence}

What is this sentence? Choose one:
- "about_me": the user tells the assistant something lasting about themself: what they like or dislike, what they avoid or cannot have, what they need, what they always or never do. It would still be true in another conversation next month.
- "reaction": the user reacts to what the assistant just said or made, or picks among things it offered.
- "task": the user describes the job at hand: what they want now, what they are working on, what happened, an error, a result.
- "text": the user is not speaking as themself here. The sentence belongs to a text they hand over to be worked on (to translate, correct, rewrite, answer, continue): a letter, an email, an essay, a story, a post, another chatbot's reply. Or it is the line of a character in a role-play or a dialogue, or the words of another person.
Answer with JSON only."""

# The prompt's own example lessons: a lesson that repeats one of them was
# copied, not written (live: Granite 3B on LM Studio gave all 9 signals the
# renaming example as their lesson).
EXAMPLE_LESSONS = (
    "When renaming a function, search for every way it is used, including names built as strings, and run the program's entry points before saying it is done.",
    "The execute tool runs JavaScript only; use the shell tool for shell commands",
    "Write commit messages in Conventional Commits style",
    "In Wordle, keep letters marked G in place",
    "When renaming a function, search for names built as strings",
)


def copied_example(lesson):
    """Whether a lesson repeats one of the prompt's examples."""
    import difflib

    text = " ".join(str(lesson or "").lower().split())
    if not text:
        return False
    if "<thing>" in text or "<other>" in text:
        return True  # a shape of USER_WISH with its placeholders left in
    return any(difflib.SequenceMatcher(None, text, " ".join(e.lower().split())).ratio() > 0.8
               or text.startswith(" ".join(e.lower().split())[:50]) for e in EXAMPLE_LESSONS)
