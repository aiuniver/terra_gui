<template>
  <div class="d-multi-select" :class="{ 'd-multi-select--active': show }" v-outside="onOutside">
    <div class="">
      <d-input-text :value="getValue" :placeholder="placeholder" @click="show = true" @clear="onClear"></d-input-text>
    </div>
    <div v-show="show" class="d-multi-select__content">
      <slot>
        <ul class="list">
          <template v-for="item of list">
            <li class="list__item" :key="item.label" @click="onSelect(item)">
              <span :class="['d-multi-select__square', { 'd-multi-select__square--active': isActive(item) }]"></span>
              {{ item.label }}
            </li>
          </template>
        </ul>
      </slot>
    </div>
  </div>
</template>

<script>
export default {
  name: 'd-multi-select',
  props: {
    value: {
      type: Array,
      default: () => [],
    },
    placeholder: {
      type: String,
      default: '',
    },
    list: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    show: false,
  }),
  computed: {
    getValue() {
      return this.value.join(', ');
    },
  },
  methods: {
    isActive({ value, label }) {
      return this.value.includes(value) || this.value.includes(label);
    },
    onOutside() {
      this.show = false;
    },
    onSelect(e) {
      this.$emit('change', e);
    },
    onClear(e) {
      this.show = !false;
      this.$emit('clear', e);
    }
  },
};
</script>

<style lang="scss">
.d-multi-select {
  &__content {
    min-width: 100%;
    position: absolute;
    left: 0;
    top: calc(100%);
    z-index: 1000;
    width: auto;
    box-shadow: 0 0.3rem 3rem 0 #36363633;
    border: none;
    border-radius: 0;
    overflow: auto;
    min-height: 4rem;
    max-height: 200px;
  }
  &__square {
    display: block;
    height: 16px;
    width: 16px;
    border: 1px solid #65b9f4;
    border-radius: 4px;
    margin-right: 10px;
    position: relative;
    &::before {
      content: '';
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      display: block;
      position: absolute;
      width: 9px;
      height: 9px;
      border-radius: 2px;
    }
    &--active {
      &::before {
        background-color: #65b9f4;
      }
    }
  }
}
</style>
