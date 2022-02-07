<template>
  <div class="d-create-toolbar">
    <div class="d-create-toolbar__dataset">
      <div class="d-create-toolbar__item" @click="handleAction('add-dataset')">
        <d-svg name="folder-add" />
      </div>
      <div class="d-create-toolbar__item" @click="handleAction('save')">
        <d-svg name="folder-save" />
      </div>
      <div class="d-create-toolbar__item" @click="handleAction('validate')">
        <d-svg name="folder-validate" />
      </div>
      <div class="d-create-toolbar__item" @click="handleAction('delete-dataset')">
        <d-svg name="folder-remove" />
      </div>
    </div>
    <div class="d-create-toolbar__workspace mt-10">
      <div class="d-create-toolbar__item d-create-toolbar__item--no-hover" @click="add({ type: 'input' })">
        <d-svg name="sloy-start-add" />
      </div>
      <div class="d-create-toolbar__item d-create-toolbar__item--no-hover" @click="add({ type: 'middle' })">
        <d-svg name="sloy-middle-add" />
      </div>
      <div class="d-create-toolbar__item d-create-toolbar__item--no-hover" @click="add({ type: 'handler' })">
        <d-svg name="sloy-handler-add" />
      </div>
      <div class="d-create-toolbar__item d-create-toolbar__item--no-hover" @click="add({ type: 'output' })">
        <d-svg name="sloy-end-add" />
      </div>
    </div>
  </div>
</template>

<script>
import { mapActions } from 'vuex';
export default {
  name: 'WorkspaceActions',
  methods: {
    ...mapActions({
      add: 'create/add',
    }),
  },
};
</script>

<style lang="scss" scoped>
@import '@/assets/scss/variables/default.scss';
.d-create-toolbar {
  position: absolute;
  top: 40%;
  left: 20px;
  transform: translateY(-50%);
  z-index: 100;
  &__item {
    margin-bottom: 10px;
    width: 50px;
    height: 50px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: var(--color-bg);
    transition: background 0.3s ease;
    cursor: pointer;
    &:hover {
      background: $color-black;
    }
    &:not(.d-create-toolbar__item--no-hover):hover {
      &::v-deep svg {
        fill: $color-light-blue;
      }
    }
    border: 1px solid $color-light-blue;
    border-radius: 2px;
  }
}
</style>