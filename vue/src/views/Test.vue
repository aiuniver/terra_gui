<template>
  <main class="page-not-found">
    <div class="cont">
      <div>
        <t-field inline>
          <t-select-new :list="list" small />
        </t-field>
      </div>
    </div>
  </main>
</template>

<script>
export default {
  name: 'page-not-found',
  data: () => ({
    list: [
      // { value: 'test', label: 'test' },
      // { value: 'dfdfdfdfdfd', label: 'tedfdfdfddddddddfdfst' },
      // { value: 'tesfd dfdft', label: 'te fdfdf dfdst' },
    ],
  }),
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
  background-color: var(--color__bg);
  .cont {
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background-color: var(--color__page);
    > div {
      width: 300px;
      height: 300px;
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