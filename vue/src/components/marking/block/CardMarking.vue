<template>
  <div class="t-card-marking">
    <div class="t-card-marking__left">
      <div class="t-card-marking__row">
        <div class="t-card-marking__content">
          <img
            src="https://upload.wikimedia.org/wikipedia/ru/thumb/5/51/%D0%92%D0%B5%D1%81%D1%82%D0%B8_%28%D0%B7%D0%B0%D1%81%D1%82%D0%B0%D0%B2%D0%BA%D0%B0%29.png/640px-%D0%92%D0%B5%D1%81%D1%82%D0%B8_%28%D0%B7%D0%B0%D1%81%D1%82%D0%B0%D0%B2%D0%BA%D0%B0%29.png"
          />
        </div>
      </div>
    </div>
    <div class="t-card-marking__center">
      <div class="t-card-marking__row">
        <div class="t-card-marking__headline">
          <h3>Разметка сегментации самолетов</h3>
        </div>
        <div class="t-card-marking__content">
          <div class="t-card-marking__item">
            <p>Задача создана</p>
            <p>17.08.2021 23:55</p>
          </div>
          <div class="t-card-marking__item">
            <p>Выполнить до</p>
            <p>21.08.2021 00:00</p>
          </div>
          <div class="t-card-marking__item">
            <p>Тип задачи</p>
            <p>Разметка изображений</p>
          </div>
        </div>
        <div class="t-card-marking__footer">
          <!-- <t-range :value="[20, 50]" /> -->
        </div>
      </div>
    </div>
    <div class="t-card-marking__right">
      <div class="t-card-marking__row">
        <div class="t-card-marking__headline">
          <h3>Назначено:</h3>
        </div>
        <div class="t-card-marking__content">
          <div class="t-card-marking__item">
            <p :style="{ backgroundColor: '#FF9D2A' }">Иван Иванов</p>
          </div>
          <div class="t-card-marking__item">
            <p :style="{ backgroundColor: '#5DBB31' }">Вова Вист</p>
          </div>
        </div>
        <div class="t-card-marking__footer">
          <button class="at-btn at-btn--default"><span class="at-btn__text">Перейти</span></button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>

export default {
  name: 't-card-marking',
  props: {
    label: String,
    type: String,
  },
  computed: {},
};
</script>

<style lang="scss" scoped>
.at-btn {
  background-color: #242f3d;
  height: 24px;
  padding: 0;
  font-size: 14px;
  &:hover {
    background-color: #2b5278;
  }
}
.t-card-marking {
  max-width: 700px;
  display: flex;
  border: 1px solid #6c7883;
  box-sizing: border-box;
  border-radius: 4px;
  background: #242f3d;
  height: 190px;
  justify-content: space-between;

  &__row {
    padding: 10px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }

  &__content {
    flex: 1 0 auto;
  }

  &__footer {
    flex: 0 0 auto;
  }

  &__headline {
    flex: 0 0 auto;
    margin-bottom: 10px;
    h3 {
      font-family: 'Open Sans';
      font-style: normal;
      font-weight: normal;
      line-height: 24px;
      font-size: 14px;
    }
  }
  p {
    font-family: 'Open Sans';
    font-style: normal;
    font-weight: normal;
    font-size: 12px;
    line-height: 24px;
  }

  &__range {
    width: 100%;
    position: relative;
    &--line {
      background: #a7bed3;
      border-radius: 3px;
      height: 6px;
      display: flex;

      &-to {
        height: 6px;
        border-radius: 3px;
      }
    }

    &--content {
      display: flex;
      justify-content: space-between;
      margin-bottom: 2px;
      p {
        color: #a7bed3;
      }
    }
  }

  &__footer {
    margin-top: 10px;
  }

  &__left {
    img {
      width: 160px;
      object-fit: cover;
      height: 100%;
      border-radius: 4px;
    }
  }

  &__center {
    min-width: 350px;
    border-right: 1px solid #17212b;
    .t-card-marking__item {
      display: flex;
      margin-top: 10px;
      &:first-child {
        margin-top: 0;
      }
      p {
        color: #a7bed3;

        &:first-child {
          flex: 0 1 120px;
        }
      }
    }
  }

  &__right {
    min-width: 170px;
    .t-card-marking__headline {
      h3 {
        color: #a7bed3;
      }
    }
    .t-card-marking__item {
      margin-top: 10px;
      p {
        color: #fff;
        padding: 2px 6px;
        border-radius: 4px;
      }
      &:first-child {
        margin-top: 0;
      }
    }
  }
}
</style>
