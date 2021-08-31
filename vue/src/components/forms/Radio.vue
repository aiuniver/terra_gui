<template>
  <div class="t-radio">
    <input style="display: none" :name="parse" :value="picked" />
    <div class="t-radio__content">
      <div class="t-radio__item" v-for="({ label, value }, i) in lists" @click="change(label)" :key="'item_' + i">
        <div :class="['t-radio__bar', { 't-radio__bar--active': active(label, value) }]"></div>
        <label class="t-radio__text">{{ label }}</label>
      </div>
    </div>

    <!-- <at-select
      class="t-field__select"
      v-model="select"
      clearable
      size="small"
      style="width: 100px"
      @on-change="change"
      :disabled="disabled"
    >
      <at-option v-for="({ label, value }, key) in items" :key="'item_' + key" :value="value">
        {{ label }}
      </at-option>
    </at-select> -->
  </div>
</template>

<script>
export default {
  name: 'TRadio',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: {
      type: String,
      default: 'radio',
    },
    value: {
      type: [String, Number],
    },
    name: {
      type: String,
      default: '',
    },
    parse: {
      type: String,
      default: '',
    },
    lists: {
      type: [Array, Object],
    },
    // disabled: Boolean,
  },
  data: () => ({
    picked: '',
    init: true,
  }),
  created() {
    const picked = this.lists.find(el => el.value);
    this.picked = picked.label;
    this.$emit('change', picked);
  },
  methods: {
    change(label) {
      this.picked = label;
      this.init = false;
      this.$emit(
        'change',
        this.lists.find(el => el.label === this.picked)
      );
    },
    active(label, value) {
      if (value && this.init) return true;
      if (this.picked && this.picked.localeCompare(label) === 0) return true;
    },
  },
};
</script>

<style lang="scss" scoped>
.t-radio {
  margin-top: 12px;
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  &__text {
    width: 150px;
    max-width: 130px;
    padding: 0 10px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }

  &__item {
    display: flex;
    align-items: center;
    margin-bottom: 14px;
  }

  &__bar {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 2px solid #6c7883;
    position: relative;
    &--active:before {
      content: '';
      display: block;
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      border-radius: 50%;
      background: #65b9f4;
      width: 10px;
      height: 10px;
    }
  }
}
</style>
