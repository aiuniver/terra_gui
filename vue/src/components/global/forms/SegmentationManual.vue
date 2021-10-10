<template>
  <div class="t-segmentation-manual">
    <t-input
      v-model="qty"
      label="Количество классов"
      type="number"
      name="classes"
      min="1"
      max="99"
      size="1"
      maxlength="2"
      :error="error"
      inline
      @focus="error = ''"
      @change="change"
    />
    <form ref="segmentation">
      <template v-for="(item, i) of +qty">
        <hr class="t-segmentation-manual__hr" :key="'hr_up' + i" />
        <t-input
          label="Название класса"
          type="text"
          :key="'classes_names_' + i"
          :parse="'classes_names[]'"
          inline
          autocomplete="off"
          @change="change"
        />
        <Color
          :value="'#ffffff'"
          label="Цвет"
          :key="'classes_colors_' + i"
          :parse="'classes_colors[]'"
          inline
          @change="change"
        />
        <hr v-if="+qty === i + 1" class="t-segmentation-manual__hr" :key="'hr_' + i" />
      </template>
    </form>
  </div>
</template>

<script>
import serialize from '@/assets/js/serialize';
import Color from '../../forms/Color.vue';
export default {
  name: 't-segmentation-manual',
  components: {
    Color,
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
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
    small: Boolean,
    // error: String,
  },
  data: () => ({
    qtyTemp: 1,
    loading: false,
    classes_names: [],
    classes_colors: [],
    error: '',
  }),
  computed: {
    qty: {
      set(value) {
        value = +value;
        if (value < 1 || value > 99) {
          this.error = 'Значение должно быть от 1 до 99';
          this.qtyTemp = 1;
        } else {
          this.qtyTemp = value;
        }
      },
      get() {
        return this.qtyTemp;
      },
    },
  },
  methods: {
    change() {
      console.log(serialize(this.$refs.segmentation));
      const { classes_names, classes_colors } = serialize(this.$refs.segmentation);
      this.$emit('change', { name: 'classes_names', value: classes_names });
      this.$emit('change', { name: 'classes_colors', value: classes_colors });
    },
  },
};
</script>

<style lang="scss" scoped>
.t-segmentation-manual {
  &__hr {
    height: 1px;
    border-width: 0;
    color: #17212b;
    background-color: #17212b;
    margin: 0 0 10px 0;
  }
}
</style>
