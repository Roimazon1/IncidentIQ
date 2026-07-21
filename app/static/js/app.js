document.addEventListener("DOMContentLoaded", () => {
  const currentUrl = new URL(window.location.href);
  if (currentUrl.searchParams.has("notice")) {
    currentUrl.searchParams.delete("notice");
    window.history.replaceState(
      {},
      "",
      `${currentUrl.pathname}${currentUrl.search}${currentUrl.hash}`,
    );
  }

  const successToast = document.getElementById("success-toast");
  if (successToast) {
    bootstrap.Toast.getOrCreateInstance(successToast).show();
  }
});
