<template>
  <div class="d-input" :class="[{ 'd-input--error': error }, { 'd-input--small': small }, { 'd-input--disabled': isDisabled }]" v-outside="outside">
    <div v-if="icon" class="d-input__icon">
      <d-svg :name="icon" />
    </div>
    <input
      v-model="input"
      v-bind="$attrs"
      autocomplete="off"
      :class="['d-input__input']"
      :type="type"
      :placeholder="placeholder"
      :disabled="isDisabled"
    />
    <div class="d-input__btn">
      <!-- <div v-show="input && !isDisabled" class="d-input__btn--cleener">
        <i class="ci-icon ci-close_big" @click="clear" />
      </div> -->
      <div :class="['d-input__btn--down', { 'd-input__btn--disabled': isDisabled }]" @click="expand">
        <i class="ci-icon ci-caret_down" />
      </div>
    </div>
    <div v-if="showContent" class="d-content">
      <scrollbar>
        <template v-for="({ label, value }, i) of list">
          <div v-if="value !== '__null__'" :class="['d-content__item', { 'd-content__item--small': small }]" :key="label + i" @click="change({ label, value })">
            {{ label }}
          </div>
        </template>
      </scrollbar>
    </div>
  </div>
</template>

<script>
import fields from '@/mixins/forms/fields';
export default {
  name: 'd-select',
  mixins: [fields],
  props: {
    list: {
      type: Array,
      default: () => [],
    },
    type: {
      type: String,
      default: 'text',
    },
    placeholder: {
      type: String,
      default: 'Введите текст',
    },
    icon: {
      type: String,
      default: '',
    },
    parse: [String, Number],
    value: [String, Number]
  },
  data: () => ({
    input: '',
    showContent: false,
  }),
  methods: {
    clear() {
      this.change({
        label: '',
        value: '',
      });
    },
    change({ label, value }) {
      this.$emit('change', { name: label, value });
      this.$emit('parse', { name: this.name, parse: this.parse, value });
      this.input = label;
      this.showContent = false;
    },
    expand() {
      this.showContent = !this.showContent;
    },
    outside() {
      this.showContent = false;
    },
  },
  created() {
    const target = this.list.find(item => item.value === this.value)
    this.input = target.label
  }
};
</script>

<style lang="scss" scoped>
@import '@/assets/scss/components/fields.scss';
.d-input {
  &__btn {
    &--down {
      height: 100%;
      width: 35px;
      display: flex;
      justify-content: center;
      align-items: center;
      border-left: 1px solid;
      border-color: inherit;
      .ci-icon {
        font-size: 20px;
      }
    }
  }
}
.d-content {
  position: absolute;
  left: 0;
  top: calc(100% + 5px);
  background-color: #242f3d;
  min-width: 100%;
  border-radius: 4px;
  overflow: hidden;
  z-index: 2;
  height: 100px;
  max-height: min-content;
  &__item {
    padding: 10px;
    cursor: pointer;
    &:hover {
      background-color: #1e2734;
      color: #65b9f4;
    }
    &--small {
      padding: 3px 10px;
      font-size: 14px;
    }
  }
}
</style>
