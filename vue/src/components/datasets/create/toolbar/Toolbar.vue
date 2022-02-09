<template>
  <div class="d-create-toolbar">
    <!-- <div class="d-create-toolbar__dataset">
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
    </div> -->
    <div class="d-create-toolbar__workspace mt-10">
      <template v-for="{ color, type, typeBlock } of filter">
        <div class="d-create-toolbar__item d-create-toolbar__item--no-hover" :key="color" @click="onAdd({ type })">
          <d-icon-layer v-bind="{ color, type, typeBlock }" />
        </div>
      </template>
    </div>
  </div>
</template>

<script>
import { mapActions, mapGetters } from 'vuex';
import { types } from '@/store/const/blocks';
export default {
  name: 'd-toolbar',
  data: () => ({
    types: types,
    toolbar: [
      { id: 3, filter: ['data', 'handler', 'input'] },
      { id: 4, filter: ['data', 'handler', 'output'] },
    ],
  }),
  computed: {
    ...mapGetters({
      getPagination: 'createDataset/getPagination',
    }),
    isActive() {
      return Boolean([3, 4].includes(this.getPagination));
    },
    filter() {
      const filter = this.toolbar.find(i => i.id === this.getPagination)?.filter || [];
      return this.types.filter(i => filter.includes(i.type));
    },
  },
  methods: {
    ...mapActions({
      add: 'create/add',
    }),
    onAdd(value) {
      if (this.isActive) {
        this.add(value);
      }
    },
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
  &--disabled {
    opacity: 0.4;
    .d-create-toolbar__item {
      cursor: default;
    }
  }
}
</style>