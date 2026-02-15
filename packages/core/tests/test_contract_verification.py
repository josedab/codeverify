"""Tests for cross-repository contract extraction and verification."""

from codeverify_core.contract_verification import (
    BreakingChangeType,
    ContractComparator,
    ContractExtractor,
    ModuleContract,
)


class TestContractExtractorPython:
    def test_extract_functions(self):
        code = '''
def create_user(name: str, email: str, age: int = 0) -> User:
    """Create a new user."""
    pass

async def delete_user(user_id: str) -> bool:
    pass

def _internal_helper():
    pass
'''
        ext = ContractExtractor()
        contract = ext.extract("python", code, "users.py")
        assert len(contract.functions) == 3

        create = next(f for f in contract.functions if f.name == "create_user")
        assert len(create.parameters) == 3
        assert create.return_type == "User"
        assert create.parameters[0].name == "name"
        assert create.parameters[2].default == "0"

        delete = next(f for f in contract.functions if f.name == "delete_user")
        assert delete.is_async is True

    def test_extract_classes(self):
        code = """
class UserService(BaseService):
    def get_user(self, user_id: str) -> User:
        pass

    def _cache_user(self, user: User) -> None:
        pass
"""
        ext = ContractExtractor()
        contract = ext.extract("python", code)
        assert len(contract.classes) == 1
        cls = contract.classes[0]
        assert cls.name == "UserService"
        assert "BaseService" in cls.bases
        assert len(cls.methods) >= 1

    def test_extract_fastapi_endpoints(self):
        code = """
@router.get("/users/{user_id}")
async def get_user(user_id: str):
    pass

@router.post("/users")
async def create_user(body: CreateUserRequest):
    pass
"""
        ext = ContractExtractor()
        contract = ext.extract("python", code)
        assert len(contract.endpoints) == 2
        paths = {e.path for e in contract.endpoints}
        assert "/users/{user_id}" in paths
        assert "/users" in paths


class TestContractExtractorGo:
    def test_extract_go_functions(self):
        code = """
func HandleRequest(ctx context.Context, req *Request) (*Response, error) {
    return nil, nil
}

func (s *Server) Start() error {
    return nil
}

func internalHelper() {
    // private
}
"""
        ext = ContractExtractor()
        contract = ext.extract("go", code)
        public = [f for f in contract.functions if f.is_public]
        assert len(public) == 2
        handle = next(f for f in contract.functions if f.name == "HandleRequest")
        assert len(handle.parameters) == 2


class TestContractExtractorJava:
    def test_extract_java_functions(self):
        code = """
public List<User> findUsers(String query, int limit) {
    return new ArrayList<>();
}

private void doInternal() {}
"""
        ext = ContractExtractor()
        contract = ext.extract("java", code)
        assert len(contract.functions) >= 1
        find = next(f for f in contract.functions if f.name == "findUsers")
        assert len(find.parameters) == 2
        assert find.return_type == "List<User>"


class TestContractComparator:
    def _make_contract(self, code: str, lang: str = "python") -> ModuleContract:
        return ContractExtractor().extract(lang, code)

    def test_detect_removed_function(self):
        old = self._make_contract("def foo(x: int) -> str:\n    pass\ndef bar() -> None:\n    pass")
        new = self._make_contract("def foo(x: int) -> str:\n    pass")

        changes = ContractComparator().compare(old, new)
        removed = [c for c in changes if c.change_type == BreakingChangeType.METHOD_REMOVED]
        assert len(removed) == 1
        assert "bar" in removed[0].entity_name

    def test_detect_param_removed(self):
        old = self._make_contract("def process(a: int, b: str) -> None:\n    pass")
        new = self._make_contract("def process(a: int) -> None:\n    pass")

        changes = ContractComparator().compare(old, new)
        assert any(c.change_type == BreakingChangeType.PARAM_REMOVED for c in changes)

    def test_detect_required_param_added(self):
        old = self._make_contract("def process(a: int) -> None:\n    pass")
        new = self._make_contract("def process(a: int, b: str) -> None:\n    pass")

        changes = ContractComparator().compare(old, new)
        assert any(c.change_type == BreakingChangeType.PARAM_ADDED_REQUIRED for c in changes)

    def test_detect_return_type_changed(self):
        old = self._make_contract("def process(a: int) -> str:\n    pass")
        new = self._make_contract("def process(a: int) -> int:\n    pass")

        changes = ContractComparator().compare(old, new)
        assert any(c.change_type == BreakingChangeType.RETURN_TYPE_CHANGED for c in changes)

    def test_no_breaking_for_optional_param_added(self):
        old = self._make_contract("def process(a: int) -> str:\n    pass")
        new = self._make_contract("def process(a: int, b: str = 'default') -> str:\n    pass")

        changes = ContractComparator().compare(old, new)
        # Adding an optional parameter is NOT breaking
        assert not any(c.change_type == BreakingChangeType.PARAM_ADDED_REQUIRED for c in changes)

    def test_detect_endpoint_removed(self):
        old_code = (
            '@router.get("/users")\n'
            "def get():\n"
            "    pass\n\n"
            '@router.post("/orders")\n'
            "def create_order():\n"
            "    pass\n"
        )
        new_code = '@router.get("/users")\ndef get():\n    pass\n'

        old = self._make_contract(old_code)
        new = self._make_contract(new_code)

        changes = ContractComparator().compare(old, new)
        assert any(c.change_type == BreakingChangeType.ENDPOINT_REMOVED for c in changes)

    def test_compatible_changes_no_breaks(self):
        old = self._make_contract("def foo(a: int) -> str:\n    pass")
        new = self._make_contract("def foo(a: int) -> str:\n    pass")

        changes = ContractComparator().compare(old, new)
        assert len(changes) == 0

    def test_breaking_change_to_dict(self):
        old = self._make_contract("def foo(a: int) -> str:\n    pass")
        new = self._make_contract("def foo(a: int) -> int:\n    pass")

        changes = ContractComparator().compare(old, new)
        assert len(changes) > 0
        d = changes[0].to_dict()
        assert "type" in d
        assert "entity" in d
        assert "description" in d
