"""Load a BehaviorTree.CPP-style mission XML into a tickable tree."""

import xml.etree.ElementTree as ET

import cognav_bt_behaviors  # noqa: F401  populates the behavior registry
from cognav_bt_behaviors.base import SEEDED_KEYS, build_behavior
from cognav_bt_behaviors.status import Status


class TreeNode:
    """Base for every node in a loaded mission tree.

    Nodes record their last status and the tick it was returned on, so the
    tick log can mark nodes that were not reached this tick.
    """

    kind = "node"
    glyph = "( )"

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__
        self.status: Status | None = None
        self.last_tick: int = -1

    @property
    def children(self) -> list["TreeNode"]:
        return []

    def tick(self, snapshot, blackboard, context) -> Status:
        raise NotImplementedError

    def _record(self, status: Status, context) -> Status:
        self.status = status
        self.last_tick = getattr(context, "tick_id", -1)
        return status


class SequenceNode(TreeNode):
    kind = "sequence"
    glyph = "[-]"

    def __init__(self, children, name: str | None = None):
        super().__init__(name or "Sequence")
        self._children = children

    @property
    def children(self) -> list[TreeNode]:
        return self._children

    def tick(self, snapshot, blackboard, context) -> Status:
        for child in self._children:
            status = child.tick(snapshot, blackboard, context)
            if status != Status.SUCCESS:
                return self._record(status, context)
        return self._record(Status.SUCCESS, context)


class FallbackNode(TreeNode):
    kind = "fallback"
    glyph = "[o]"

    def __init__(self, children, name: str | None = None):
        super().__init__(name or "Fallback")
        self._children = children

    @property
    def children(self) -> list[TreeNode]:
        return self._children

    def tick(self, snapshot, blackboard, context) -> Status:
        for child in self._children:
            status = child.tick(snapshot, blackboard, context)
            if status != Status.FAILURE:
                return self._record(status, context)
        return self._record(Status.FAILURE, context)


class RetryNode(TreeNode):
    """Re-tick one child within a single tick until it succeeds.

    Every attempt reads the same snapshot, so a different outcome can only come
    from a different proposal; the safety gate passes the bins it rejected to
    the planner through `rejected_bins`.
    """

    kind = "decorator"
    glyph = "[R]"

    def __init__(self, child, num_attempts: int, name: str | None = None):
        super().__init__(name or "Retry")
        self._child = child
        self._num_attempts = num_attempts
        #: Attempts consumed on the most recent tick, for the tick log.
        self.attempts_used = 0

    @property
    def children(self) -> list[TreeNode]:
        return [self._child]

    def tick(self, snapshot, blackboard, context) -> Status:
        status = Status.FAILURE
        for attempt in range(self._num_attempts):
            self.attempts_used = attempt + 1
            status = self._child.tick(snapshot, blackboard, context)
            if status == Status.SUCCESS:
                break
        return self._record(status, context)


class BehaviorNode(TreeNode):
    kind = "leaf"
    glyph = "-->"

    def __init__(self, behavior):
        super().__init__(behavior.name)
        self.behavior = behavior

    def tick(self, snapshot, blackboard, context) -> Status:
        return self._record(self.behavior.tick(snapshot, blackboard, context), context)


def load_tree(xml_text: str) -> TreeNode:
    root = ET.fromstring(xml_text)
    tree_id = root.attrib.get("main_tree_to_execute")
    behavior_tree = None
    for candidate in root.findall("BehaviorTree"):
        if tree_id is None or candidate.attrib.get("ID") == tree_id:
            behavior_tree = candidate
            break
    if behavior_tree is None:
        raise ValueError("No matching BehaviorTree found in mission XML")
    children = list(behavior_tree)
    if len(children) != 1:
        raise ValueError("BehaviorTree must contain exactly one root control node")
    root = _parse_node(children[0])
    validate_contracts(root)
    return root


def validate_contracts(root: TreeNode) -> None:
    """Reject a tree in which a leaf reads a key nothing has written yet.

    The walk is pre-order, which is tick order for these composites. Keys in
    READS_CARRIED are skipped, and a key written anywhere earlier in the walk
    counts as available even if its branch does not run on every tick.
    """
    available = set(SEEDED_KEYS)
    problems: list[str] = []

    def walk(node: TreeNode) -> None:
        behavior = getattr(node, "behavior", None)
        if behavior is not None:
            missing = sorted(set(behavior.READS) - available)
            if missing:
                problems.append(
                    f"  <{behavior.__class__.__name__} name='{node.name}'> reads "
                    f"{', '.join(missing)} before anything writes {'it' if len(missing) == 1 else 'them'}"
                )
            available.update(behavior.WRITES)
        for child in node.children:
            walk(child)

    walk(root)
    if problems:
        raise ValueError(
            "mission tree violates the role contracts:\n" + "\n".join(problems)
            + "\n(see cognav_bt_behaviors/base.py for what each role reads and writes)"
        )


def _parse_node(element) -> TreeNode:
    tag = _strip_namespace(element.tag)
    children = [_parse_node(child) for child in list(element)]
    name = element.attrib.get("name")
    if tag in ("Sequence", "ReactiveSequence"):
        return SequenceNode(children, name=name)
    if tag in ("Fallback", "Selector", "ReactiveFallback"):
        return FallbackNode(children, name=name)
    if tag in ("Retry", "RetryUntilSuccessful"):
        if len(children) != 1:
            raise ValueError(
                f"<{tag}> takes exactly one child, got {len(children)}; "
                f"wrap several nodes in a Sequence"
            )
        raw = element.attrib.get("num_attempts", "3")
        try:
            attempts = int(raw)
        except ValueError:
            raise ValueError(f"<{tag} num_attempts='{raw}'> is not an integer") from None
        if attempts < 1:
            raise ValueError(f"<{tag} num_attempts='{attempts}'> must be at least 1")
        return RetryNode(children[0], attempts, name=name)
    if children:
        raise ValueError(f"Unknown control node '{tag}'")
    # Attributes other than `name` are ports, validated against BehaviorConfig.
    return BehaviorNode(build_behavior(tag, name, element.attrib))


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag
