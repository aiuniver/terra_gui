<template>
  <main class="page-not-found">
    <div class="cont">
      <i class="t-icon icon-logo"></i>
      <h1 class="title">Страница <span>{{ url }}</span> не найдена</h1>
      <span class="title">
        <router-link to="/">На главную</router-link>
        </span>
    </div>
  </main>
</template>

<script>
export default {
  name: 'page-not-found',
  computed: {
    url() {
      return decodeURI(window.location.pathname);
    },
  },
  created() {
    console.log('sdsdsds');
    // window.location.href = "/";
  },
};
</script>

<style lang="scss" scoped>
.page-not-found {
  width: 100%;
  height: 100%;
  background-color: var(--color-bg);
  .cont {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background-color: var(--color-page);
    .t-icon{
      height: 70px;
      width: 70px;
      margin-bottom: 20px;
    }
  }
}
.title {
    color: #6c7883;
    font-size: 1.5rem;
    span {
      color: #a9a9a9;
      font-weight: 700;
    }
}
</style>