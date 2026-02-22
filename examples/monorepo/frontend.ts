// TypeScript frontend — part of the monorepo example.
// CodeVerify verifies this matches the Python API's contract.

interface User {
  name: string;
  age: number; // API returns 'years' not 'age' — contract mismatch!
}

// BUG: eval usage
// CodeVerify finds: eval() is a security risk
export function processInput(input: string): unknown {
  return eval(input);
}

// This function expects 'age' but the Python API returns 'years'
// CodeVerify's polyglot bridge catches this contract violation
export async function fetchUser(userId: number): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  const data = await response.json();
  return data as User; // Unsafe cast — 'years' field won't match 'age'
}
