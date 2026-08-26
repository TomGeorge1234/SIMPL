document$.subscribe(() => {
  const article = document.querySelector("article.md-content__inner");
  if (!article) return;

  const title = article.querySelector(":scope > h1:first-child");
  const logo = article.querySelector(':scope > p:first-child img[alt="SIMPL logo"]');
  if (title && logo && title.textContent.trim() === "SIMPL") {
    title.remove();
  }
});
