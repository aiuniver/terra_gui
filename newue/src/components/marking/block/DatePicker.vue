<template>
  <label class="t-datepicker">
    <date-picker
      type="datetime"
      format="DD.MM.YYYY HH:mm"
      v-model="date"
      :lang="lang"
      @change="change"
      value-type="DD.MM.YYYY HH:mm"
      :editable="false"
    ></date-picker>
    <span class="t-datepicker__label">
      {{ label }}
    </span>
  </label>
</template>

<script>
import DatePicker from '@/assets/js/datepicker.min.js';
import '@/assets/css/datepicker/datepicker.scss';

export default {
  name: 't-datepicker',
  components: {
    DatePicker,
  },
  props: ['label', 'value'],
  data: () => ({
    date: '',
    lang: {
      formatLocale: {
        months: ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'],
        weekdaysMin: ['Вс', 'Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб'],
        monthsShort: ['ЯНВ', 'ФЕВ', 'МАР', 'АПР', 'МАЙ', 'ИЮНЬ', 'ИЮЛЬ', 'АВГ', 'СЕН', 'ОКТ', 'НОЯ', 'ДЕК'],
        firstDayOfWeek: 1,
      },
      monthBeforeYear: true,
    },
  }),
  methods: {
    change(e) {
      this.$emit('input', e);
    },
  },
};
</script>

<style lang="scss">
.t-datepicker {
  &__label {
    font-size: 12px;
    line-height: 12px;
    color: #a7bed3;
    padding: 0 10px;
    display: inline-block;
  }
}
</style>