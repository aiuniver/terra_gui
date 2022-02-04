<template>
  <div class="dataset-card-new" @click="onClick">
    <div class="card-borders"></div>
    <div class="dataset-card-new__wrapper">
      <p class="dataset-card-new__title">{{ dataset.name }}</p>
      <div class="dataset-card-new__info">
        <p>login_name_user</p>
        <div class="dataset-card-new__info--size">
          <i class="ci-icon ci-image" />
          <span>{{ dataset.size ? dataset.size.short.toFixed(2) + ' ' + dataset.size.unit : 'Предустановленный' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name:"DatasetCard",
  props: {
    dataset: {
      type: Object,
      default: () => {},
    },
  },
  methods: {
    onClick() {
      this.$emit('click', this.dataset)
    },
  },
};
</script>

<style lang="scss">
.dataset-card-new {
  width: 181px;
  height: 141px;
  position: relative;
  background-color: #1B2935;
  cursor: pointer;
  mask-image: url("data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ic3ZnLWJvcmRlciIgd2lkdGg9IjE4MSIgaGVpZ2h0PSIxNDEiIHZpZXdCb3g9IjAgMCAxODEgMTQxIiBmaWxsPSIjMUIyOTM1IiBmaWxsLXJ1bGU9Im5vbnplcm8iIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgICA8cGF0aCBkPSJNMTI2LjUgMTE3LjVIMTI2LjMwOEwxMjYuMTY1IDExNy42MjlMMTAxLjgwNyAxMzkuNTk5QzEwMS4xNjQgMTQwLjE3OSAxMDAuMzI5IDE0MC41IDk5LjQ2MjYgMTQwLjVIMjlINEMyLjA2NyAxNDAuNSAwLjUgMTM4LjkzMyAwLjUgMTM3VjRDMC41IDIuMDY3IDIuMDY3IDAuNSA0IDAuNUgxNTguOTMxQzE1OS44MTQgMC41IDE2MC42NjMgMC44MzMyMzMgMTYxLjMxMSAxLjQzMjk5TDE3OS4zNzkgMTguMTc5NUMxODAuMDk0IDE4Ljg0MTkgMTgwLjUgMTkuNzcyMSAxODAuNSAyMC43NDY1VjExNy41SDEyNi41WiIgc3Ryb2tlPSIjMUIyOTM1Ii8+CiAgICA8c3ZnIHg9IjExOCIgeT0iMTI1IiB3aWR0aD0iNjMiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDYzIDkiIGZpbGw9IiMxQjI5MzUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CiAgICAgICAgPHBhdGggZD0iTTEwLjY4NSAwLjVINjIuNVY1QzYyLjUgNi45MzMgNjAuOTMzIDguNSA1OSA4LjVIMS4zNTE2M0wxMC42ODUgMC41WiIgc3Ryb2tlPSIjMUIyOTM1Ii8+CiAgICA8L3N2Zz4KPC9zdmc+Cg==");
  &:hover {
    background-color: #253848;
  }
  &__wrapper {
    padding: 20px 15px 10px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
  }
  &__title {
    font-size: 14px;
    line-height: 24px;
  }
  &__info {
    color: #828282;
    font-size: 11px;
    * {
      font-size: inherit;
      color: inherit;
    }
    i {
      font-size: 15px;
      margin-right: 9px;
    }
    &--size {
      max-width: 100px;
      text-overflow: ellipsis;
      overflow: hidden;
      white-space: nowrap;
      margin-top: 10px;
      * {
        vertical-align: middle;
      }
    }
  }
}

.card-borders {
  position: absolute;
  width: 100%;
  height: 100%;
  mask-image: url('data:image/svg+xml;base64,PHN2ZyBjbGFzcz0ic3ZnLWJvcmRlciIgd2lkdGg9IjE4MSIgaGVpZ2h0PSIxNDEiIHZpZXdCb3g9IjAgMCAxODEgMTQxIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgPHBhdGggZD0iTTEyNi41IDExNy41SDEyNi4zMDhMMTI2LjE2NSAxMTcuNjI5TDEwMS44MDcgMTM5LjU5OUMxMDEuMTY0IDE0MC4xNzkgMTAwLjMyOSAxNDAuNSA5OS40NjI2IDE0MC41SDI5SDRDMi4wNjcgMTQwLjUgMC41IDEzOC45MzMgMC41IDEzN1Y0QzAuNSAyLjA2NyAyLjA2NyAwLjUgNCAwLjVIMTU4LjkzMUMxNTkuODE0IDAuNSAxNjAuNjYzIDAuODMzMjMzIDE2MS4zMTEgMS40MzI5OUwxNzkuMzc5IDE4LjE3OTVDMTgwLjA5NCAxOC44NDE5IDE4MC41IDE5Ljc3MjEgMTgwLjUgMjAuNzQ2NVYxMTcuNUgxMjYuNVoiIHN0cm9rZT0iIzY1QjlGNCIvPgogICAgPHN2ZyB4PSIxMTgiIHk9IjEyNSIgd2lkdGg9IjYzIiBoZWlnaHQ9IjkiIHZpZXdCb3g9IjAgMCA2MyA5IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogICAgICAgIDxwYXRoIGQ9Ik0xMC42ODUgMC41SDYyLjVWNUM2Mi41IDYuOTMzIDYwLjkzMyA4LjUgNTkgOC41SDEuMzUxNjNMMTAuNjg1IDAuNVoiIHN0cm9rZT0iIzY1QjlGNCIvPgogICAgPC9zdmc+Cjwvc3ZnPgo=');
  background-color: #65B9F4;
  clip-path: polygon(0% 0%, 0 40px, 100% 40px, 100% 95px, 0 95px, 0 100%, 15px 100%, 15px 130px, 80px 130px, 80px 100%, 100% 100%, 100% 0);
  z-index: -1;
}
</style>