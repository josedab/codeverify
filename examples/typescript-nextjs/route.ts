// Example Next.js API route with intentional issues for CodeVerify.

import { NextRequest, NextResponse } from "next/server";

// BUG: Uses 'any' type — bypasses type safety
// CodeVerify finds: use of 'any' bypasses type checking
export async function GET(request: NextRequest): Promise<NextResponse> {
  const data: any = await fetchData();
  return NextResponse.json({ result: data.value });
}

// BUG: eval() usage — code injection risk
// CodeVerify finds: eval() is a security risk
export async function POST(request: NextRequest): Promise<NextResponse> {
  const body = await request.json();
  const result = eval(body.expression);
  return NextResponse.json({ result });
}

// BUG: XSS via innerHTML (simulated)
function renderUserContent(html: string): string {
  // CodeVerify finds: unsanitized HTML rendering
  return `<div>${html}</div>`;
}

async function fetchData(): Promise<unknown> {
  return { value: 42 };
}
