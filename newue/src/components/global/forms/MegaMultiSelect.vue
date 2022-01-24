<template>
  <div class="t-mega-select">
    <div class="t-mega-select__header">{{ label }}</div>
    <div :class="['t-mega-select__body', { 't-mega-select__body--disabled': disabled }]">
      <scrollbar :ops="ops">
        <div v-for="({ label, value }, i) of list" :key="'mega_' + i" class="t-mega-select__list" @click="click(value)">
          <div :class="['t-mega-select__list--switch', { 't-mega-select__list--active': isActive(value) }]">
            <span></span>
          </div>
          <div class="t-mega-select__list--label">{{ label }}</div>
        </div>
      </scrollbar>
      <div v-show="!list.length" class="t-mega-select__body--empty">Нет данных</div>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-multu-select',
  props: {
    label: String,
    value: Array,
    name: String,
    parse: String,
    list: {
      type: Array,
      default: () => [],
    },
    disabled: Boolean,
  },
  data: () => ({
    valueTemp: [],
    ops: {
      bar: { background: '#17212b' },
      scrollPanel: {
        scrollingX: false,
        scrollingY: true,
      },
    },
  }),
  methods: {
    isActive(value) {
      return this.valueTemp.includes(value);
    },
    click(value) {
      if (!this.disabled) {
        if (this.valueTemp.length > 1 || !this.valueTemp.includes(value)) {
          this.valueTemp = this.valueTemp.includes(value)
            ? this.valueTemp.filter(item => item !== value)
            : [...this.valueTemp, value];
          this.$emit('input', this.valueTemp);
          this.$emit('change', { name: this.name, value });
          this.$emit('parse', { name: this.name, parse: this.parse, value: this.valueTemp });
        }
      }
    },
  },
  created() {
    this.valueTemp = this.value;
    // console.log(this.valueTemp);
  },
};
</script>

<style lang="scss" scoped>
.t-mega-select {
  margin: 10px 0 10px 0;
  &__header {
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1;
    font-size: 0.75rem;
    user-select: none;
  }
  &__body {
    border: 1px solid #65B9F4;
    border-radius: 4px;
    padding: 5px 0 5px 5px;
    overflow: hidden;
    background: #242f3d;
    height: 100px;
    &--empty {
      color: #a7bed3;
      font-size: 0.7em;
      line-height: 1em;
      padding: 0 8px;
      text-decoration: none;
      display: block;
      cursor: default;
    }
    &--disabled {
      opacity: 0.3;
    }
  }
  &__list {
    display: flex;
    // margin-bottom: 2px;
    padding: 1px 0;
    align-items: center;
    overflow: hidden;
    &--label {
      user-select: none;
      width: auto;
      padding: 0 0 0 10px;
      text-align: left;
      color: #a7bed3;
      display: block;
      margin: 0;
      font-size: 0.75rem;
      text-overflow: ellipsis;
      overflow: hidden;
      cursor: default;
    }
    &--switch {
      flex: 0 0 35px;
      height: 20px;
      position: relative;
      span {
        background-color: #6C7883;
        display: block;
        position: relative;
        height: 100%;
        border-radius: 20px;
        transition: 0.2s;
        cursor: pointer;
        &:before {
          background-color: #0E1621;
          display: block;
          content: '';
          height: 16px;
          width: 16px;
          position: absolute;
          left: 2px;
          top: 2px;
          border-radius: 50%;
          transition: 0.2s;
        }
      }
    }
    &--active {
      span {
        background-color: #65B9F4;
      }
      span:before {
        transform: translateX(15px);
      }
    }
  }
}
</style>
