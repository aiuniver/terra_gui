<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" @click="clickLabel">
      <slot>{{ label }}</slot>
    </label>
    <div class="t-field__switch">
      <input
        v-model="checVal"
        class="t-field__input"
        :checked="checVal ? 'checked' : ''"
        type="checkbox"
        :value="checVal"
        :name="parse"
        :data-reverse="reverse"
        @change="change"
      />
      <span></span>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-checkbox',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    inline: Boolean,
    value: Boolean,
    name: String,
    parse: String,
    reverse: Boolean,
    event: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    checVal: false,
  }),
  methods: {
    change(e) {
      const value = e.target.checked;
      this.$emit('change', { name: this.name, value });
      if (this.error) {
        this.$emit('cleanError', true);
      }
    },
    clickLabel() {
      this.checVal = !this.checVal;
      this.$emit('change', { name: this.name, value: this.checVal });
    },
  },
  created() {
    this.checVal = this.value;
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 20px;
  &__label {
    width: 150px;
    max-width: 330px;
    padding-bottom: 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    // white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    width: 100%;
    height: 100%;
    position: absolute;
    left: 0;
    top: 0;
    z-index: 1;
    opacity: 0;
    cursor: pointer;
    &:focus {
      border-color: #fff;
    }
    &:checked + span:before {
      transform: translateX(12px);
      background-color: #65b9f4;
    }
  }
  &__switch {
    width: 26px;
    height: 14px;
    position: relative;

    span {
      background-color: #242f3d;
      border-color: #6c7883 !important;
      display: block;
      position: relative;
      height: 100%;
      border: 1px solid;
      border-radius: 4px;
      transition: 0.2s;
      cursor: pointer;
      &:before {
        background-color: #6c7883;
        display: block;
        content: '';
        height: 10px;
        width: 10px;
        position: absolute;
        left: 1px;
        top: 1px;
        border-radius: 2px;
        transition: 0.2s;
      }
    }
  }
}

.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;

  > label {
    width: auto;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    font-size: 0.75rem;
  }
  > input {
    width: 100%;
    height: 100%;
    position: absolute;
    left: 0;
    top: 0;
    z-index: 1;
    opacity: 0;
    cursor: pointer;
  }
}
</style>
