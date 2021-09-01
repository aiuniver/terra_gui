<template>
  <div class="t-segmentation-annotation">
    <div :class="['t-inline']">
      <label class="t-field__label"><slot></slot></label>
      <t-button class="t-field__button" :disabled="disabled" @click.native="getApi" :loading="loading">Найти</t-button>
    </div>
    <template v-for="({ name, color }, i) of items">
      <hr class="t-segmentation-annotation__hr" :key="'hr_up' + i" />
      <t-input
        :value="name"
        label="Название класса"
        type="text"
        name="classes_names"
        :key="'classes_names_' + i"
        :parse="'classes_names[]'"
        inline
        disabled
        @change="change"
      />
      <Color :value="color" label="Цвет" :key="'classes_colors_' + i" inline disabled />
      <hr v-if="items.length === i + 1" class="t-segmentation-annotation__hr" :key="'hr_' + i" />
    </template>
  </div>
</template>

<script>
import Color from '../../forms/Color.vue';
export default {
  name: 't-segmentation-annotation',
  components: {
    Color
  },
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
      type: [String, Number],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
    small: Boolean,
    error: String,
  },
  data: () => ({
    loading: false,
    items: []
  }),
  computed: {},
  methods: {
    async getApi() {
      if (this.loading) return; 
      this.loading = true
      const { data } = await this.$store.dispatch('datasets/classesAnnotation')
      console.log(data)
      this.items = data
      this.loading = false
      const classes_names = data.map(item => item.name)
      const classes_colors = data.map(item => item.color)
      this.$emit('change', { name: 'classes_names', value: classes_names } );
      this.$emit('change', { name: 'classes_colors', value: classes_colors } );
    },
    change() {
      // if (this.isChange) {
      //   let value = e.target.value;
      //   value = this.type === 'number' ? +value : value;
      //   this.$emit('change', { name: this.name, value });
      //   this.isChange = false;
      // }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  // align-items: center;
  .t-field__label {
    padding: 6px 0 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
  }
  .t-field__button {
    flex: 0 0 100px;
    height: 24px;
    font-size: 12px;
    line-height: 24px;
  }
}
.t-segmentation-annotation {
  &__hr {
    height: 1px;
    border-width: 0;
    color: #17212b;
    background-color: #17212b;
    margin: 0 0 10px 0;
  }
}
</style>
