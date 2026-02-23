# CodeVerify Launch Checklist

Everything needed to go from "built" to "shipped". All infrastructure
code is committed — these are operational steps only.

## Phase 1: Publish (Day 1)

- [ ] **Push release tag**
  ```bash
  git tag v1.6.0
  git push origin v1.6.0
  ```
  This triggers `.github/workflows/publish.yml` which publishes:
  - `codeverify-core` to PyPI
  - `codeverify-verifier` to PyPI
  - `codeverify-agents` to PyPI
  - VS Code extension to VS Code Marketplace
  - `@codeverify/widget` to npm
  - GitHub Release with changelog

- [ ] **Verify packages are live**
  ```bash
  pip install codeverify-core    # Should work within 10 min
  pip install codeverify-verifier
  pip install codeverify-agents
  ```

## Phase 2: Deploy (Day 1-2)

- [ ] **Create K8s cluster** (AWS EKS, GCP GKE, or Azure AKS)

- [ ] **Create secrets**
  ```bash
  kubectl create secret generic codeverify-secrets \
    --from-literal=db-user=codeverify \
    --from-literal=db-password=$(openssl rand -base64 32) \
    --from-literal=database-url=postgresql://... \
    --from-literal=jwt-secret=$(openssl rand -base64 32) \
    --from-literal=stripe-secret=sk_live_... \
    --from-literal=openai-key=sk-... \
    --from-literal=anthropic-key=sk-ant-...
  ```

- [ ] **Deploy**
  ```bash
  kubectl apply -f deploy/kubernetes/production.yml
  ```

- [ ] **Configure DNS**
  - `codeverify.dev` → web service ingress
  - `api.codeverify.dev` → API service ingress

- [ ] **Configure Stripe**
  - Create products: Free ($0), Pro ($49/mo), Enterprise ($199/mo)
  - Set up webhooks → `api.codeverify.dev/webhooks/stripe`

- [ ] **Or use one-command deploy**
  ```bash
  python scripts/deploy_cloud.py --provider aws --domain codeverify.dev
  ```

## Phase 3: Community (Day 2)

- [ ] **Enable GitHub Discussions**
  - Repository → Settings → Features → ✅ Discussions
  - Templates are already committed in `.github/DISCUSSION_TEMPLATE/`

- [ ] **Create Discord server**
  - Follow channel structure in `COMMUNITY.md`
  - Channels: #general, #help, #show-and-tell, #contributing, #z3-verification, #ai-agents, #announcements
  - Add invite link to README.md and COMMUNITY.md

- [ ] **Submit GitHub Marketplace listing**
  - Use content from `docs/github-marketplace-listing.md`
  - Take screenshots of: PR review, dashboard, VS Code extension, proof debugger
  - Submit for review

## Phase 4: Content (Day 3-5)

- [ ] **Record demo video** (3 minutes)
  - Install GitHub App → Open PR → See results → Click fix → Proof debugger
  - Publish to YouTube
  - Embed in README.md

- [ ] **Launch on Hacker News**
  - Title: "Show HN: CodeVerify — AI code review with Z3 formal verification"
  - Post at 9am PT on a Tuesday
  - Monitor and respond to all comments for 24 hours

## Phase 5: Verify

- [ ] `pip install codeverify-core` works
- [ ] `codeverify.dev` loads
- [ ] GitHub App installable from Marketplace
- [ ] VS Code extension installable from Marketplace
- [ ] Discord server has 10+ members
- [ ] First external PR reviewed by CodeVerify

---

## Estimated Timeline

| Day | Milestone |
|-----|-----------|
| 1 | Tag pushed, packages on PyPI, K8s deployed |
| 2 | DNS live, Stripe configured, Discussions enabled, Discord created |
| 3 | Marketplace submitted, demo video recorded |
| 5 | HN launch |
| 7 | First 100 PyPI installs, 50 GitHub App installs |
| 30 | First external contributor |
