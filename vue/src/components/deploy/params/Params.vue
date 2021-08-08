<template>
  <div class="params">
    <scrollbar :style="height">
      <div class="params-container__name">Загрузка в демо-панель</div>
      <div class="params-container pa-5">
        <div class="label">Название папки</div>
        <t-input :label="'https://demo.neural-university.ru/login/project/'" :labelColor="'#6C7883'"/>
        <Checkbox :label="'Перезаписать с таким же названием папки'" :type="'checkbox'" class="pd__top"/>
        <Checkbox :label="'Использовать пароль для просмотра страницы'" :type="'checkbox'"/>
        <div class="t-input">
          <input type="password" placeholder="Введите пароль">
        </div>
        <div class="t-input">
          <input type="password" placeholder="Подтверждение пароля">
        </div>
        <div class="loader">
          <div class="loader__title">Дождитесь окончания загрузки</div>
          <div class="loader__time">
            Загружено 892 MB из 1.2 GB    Осталось: меньше минуты
          </div>
          <div class="loader__progress">
            <div class="progress-bar">
              <div class="loading">
                <span>78%</span>
              </div>
            </div>
          </div>
        </div>
        <button>Отправить</button>
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
import TInput from "@/components/forms/TInput.vue";
import Checkbox from "@/components/forms/Checkbox";
export default {
  name: "Settings",
  components: {
    TInput,
    Checkbox
  },
  data: () => ({}),
  computed: {
  ...mapGetters({
    height: "settings/height",
  }),
  },
  methods: {
    click() {
      console.log();
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  flex-shrink: 0;
  width: 470px;
  border-left: #0e1621 solid 1px;
}
.params-container__name{
  padding: 30px 0 0 20px;
}
.pd__top{
  padding: 10px 0;
}
.label{
  color: #A7BED3;
  font-size: 12px;
  line-height: 24px;
}
.color__grey{
  color: #6C7883;
}
button {
  font-size: 0.875rem;
  margin-top: 10px;
  width: 107px;
}
.t-input{
  padding-bottom: 10px;
}
.loader__title{
  font-size: 14px;
  line-height: 24px;
}
.loader__time{
  padding-top: 20px;
  font-size: 12px;
  line-height: 24px;
  color: #A7BED3;
  width: 100%;
  text-align: center;
}
.loader__progress{
  padding-top: 10px;
  width: 100%;
  display: flex;
  justify-content: center;
}
.progress-bar{
  width: 426px;
  background: #2B5278;
  border-radius: 4px;
}
.loading{
  background: #65B9F4;
  border-radius: 4px;
  width: 78%;
  span{
    position: relative;
    margin-left: 201px;
  }
}
</style>