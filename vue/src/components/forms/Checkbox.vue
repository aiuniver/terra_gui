<template>
  <div class="t-field t-inline">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <div class="t-field__switch">
      <input
        :id="parse"
        class="t-field__input"
        :checked="checked ? 'checked' : ''"
        :type="type"
        :value="checked"
        :name="parse"
        @change="change"
      />
      <span></span>
    </div>
  </div>
</template>

<script>
import { bus } from '@/main';
export default {
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: {
      type: String,
      default: 'text',
    },
    value: {
      type: [Boolean],
    },
    name: {
      type: String,
    },
    parse: {
      type: String,
    },
    event: {
      type: Array,
      default: () => [],
    },
  },
  data: () => ({
    checked: null,
  }),
  methods: {
    change(e) {
      // console.log(e);
      const value = e.target.checked;
      this.$emit('change', { name: this.name, value });
      bus.$emit('change', { event: this.name, value });
    },
  },
  created() {
    this.checked = this.value;
    // console.log('created ' + this.name, this.checked)
    if (this.event.length) {
      // console.log("created", this.name);
      bus.$on('change', ({ event }) => {
        if (this.event.includes(event)) {
          this.checked = false;
        }
      });
    }
  },
  destroyed() {
    if (this.event.length) {
      bus.$off();
      console.log('destroyed', this.name);
    }
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 20px;
  &__label {
    width: 150px;
    max-width: 330px;
    padding: 0 10px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
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
    padding: 0 20px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
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
