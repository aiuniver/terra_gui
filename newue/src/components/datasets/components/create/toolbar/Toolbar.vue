<template>
  <div class="page-datasets-action">
    <div class="page-datasets-action__dataset">
      <div class="page-datasets-action__item" @click="handleAction('add-dataset')">
        <SvgContainer name="folder-add" />
      </div>
      <div class="page-datasets-action__item" @click="handleAction('save')">
        <SvgContainer name="folder-save" />
      </div>
      <div class="page-datasets-action__item" @click="handleAction('validate')">
        <SvgContainer name="folder-validate" />
      </div>
      <div class="page-datasets-action__item" @click="handleAction('delete-dataset')">
        <SvgContainer name="folder-remove" />
      </div>
    </div>
    <div class="page-datasets-action__workspace mt-10">
      <div class="page-datasets-action__item page-datasets-action__item--no-hover" @click="handleAction('add-input')">
        <SvgContainer name="sloy-start-add" />
      </div>
      <div class="page-datasets-action__item page-datasets-action__item--no-hover">
        <SvgContainer name="sloy-middle-add" @click="handleAction('add-middle')" />
      </div>
      <div class="page-datasets-action__item page-datasets-action__item--no-hover" @click="handleAction('add-end')">
        <SvgContainer name="sloy-end-add" />
      </div>
    </div>
  </div>
</template>

<script>

export default {
  name: 'WorkspaceActions',
  methods: {
    handleAction(action) {
      this.$emit('action', action);
    },
  },
};
</script>

<style lang="scss" scoped>
.page-datasets {
  &-action {
    position: absolute;
    top: 50%;
    left: 20px;
    transform: translateY(-50%);
    &__item {
      margin-bottom: 10px;
      width: 50px;
      height: 50px;
      display: flex;
      justify-content: center;
      align-items: center;
      background: transparent;
      transition: background 0.3s ease;
      cursor: pointer;
      &:hover {
        background: $color-black;
      }
      &:not(.page-datasets-action__item--no-hover):hover {
        &::v-deep svg {
          fill: $color-light-blue;
        }
      }
      border: 1px solid $color-light-blue;
      border-radius: 2px;
    }
  }
}
</style>