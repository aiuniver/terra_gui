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
        <div class="load__answer">
          <scrollbar>
            <div class="terra-url">https://srv2.demo.neural-university.ru</div>
            <div class="answer-url">
              Чтобы получилось:
              <span>
              https://srv2.demo.neural-university.ru/media/check_od/output_files/34234452.jpg
            </span>
            </div>
            <div class="module-list">
              <div class="module-list__title">СПИСОК МОДУЛЕЙ:</div>
              <div class="module-list__item">
                <div class="item__title">Распознавание птиц на видео</div>
                <div class="item__url">https://demo.neural-university.ru/birds_video/</div>
                <div class="item__input">
                  Вход:
                  <span>od_content файл с изображением</span>
                </div>
                <div class="item__output">
                  Выход:
                  <span>Относительная ссылка</span>
                </div>
              </div>
              <div class="module-list__item">
                <div class="item__title">Распознавание птиц на видео</div>
                <div class="item__url">https://demo.neural-university.ru/birds_video/</div>
                <div class="item__input">
                  Вход:
                  <span>od_content файл с изображением</span>
                </div>
                <div class="item__output">
                  Выход:
                  <span>Относительная ссылка</span>
                </div>
              </div>
              <div class="module-list__item">
                <div class="item__title">Распознавание птиц на видео</div>
                <div class="item__url">https://demo.neural-university.ru/birds_video/</div>
                <div class="item__input">
                  Вход:
                  <span>od_content файл с изображением</span>
                </div>
                <div class="item__output">
                  Выход:
                  <span>Относительная ссылка</span>
                </div>
              </div>
            </div>
          </scrollbar>
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
.load__answer{
  height: 464px;
  padding: 16px;
  margin-top: 20px;
  background: #242F3D;
  border: 1px solid #6C7883;
  box-sizing: border-box;
  border-radius: 4px;
}
.terra-url{
  color: #65B9F4;
  font-size: 14px;
  line-height: 21px;
  font-style: italic;
  padding-bottom: 20px;
}
.answer-url{
  font-size: 14px;
  line-height: 21px;
  span{
    padding-top: 20px;
    display: block;
    font-size: 14px;
    line-height: 21px;
    font-style: italic;
    color: #65B9F4;
  }
}
.module-list{
  padding-top: 40px;
}
.module-list__title{
  font-size: 16px;
  line-height: 21px;
  font-weight: 700;
  padding-bottom: 10px;
}
.module-list__item{
  padding: 10px 0;
}
.item__title{
  font-size: 14px;
  line-height: 21px;
  font-weight: 700;
}
.item__url{
  color: #65B9F4;
  font-size: 14px;
  line-height: 21px;
  font-style: italic;
}
.item__input, .item__output{
  span{
    display: block;
  }
}

</style>