#!/usr/bin/env python3
#
# version1:  1401/05/04
# version2:  1405/03/26  (Using ChatGPT)
# writen by: Iman Kahbasi


from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import date, datetime
import jdatetime


@dataclass
class DateDetails:
    date: str
    year: int
    week: int
    doy: int
    weekday: str


def get_details(dt_obj) -> DateDetails:
    """
    Extract calendar information from a date object.
    """

    iso_year, week, _ = dt_obj.isocalendar()

    return DateDetails(
        date=str(dt_obj),
        year=dt_obj.year,
        week=week,
        doy=dt_obj.timetuple().tm_yday,
        weekday=dt_obj.strftime("%A"),
    )


def jalali_to_gregorian(date_str):
    """
    Convert Jalali date string to Gregorian date.
    """

    jalali = jdatetime.datetime.strptime(
        date_str,
        "%Y-%m-%d"
    ).date()

    gregorian = jalali.togregorian()

    return gregorian, jalali


def gregorian_to_jalali(date_str):
    """
    Convert Gregorian date string to Jalali date.
    """

    gregorian = datetime.strptime(
        date_str,
        "%Y-%m-%d"
    ).date()

    jalali = jdatetime.date.fromgregorian(
        date=gregorian
    )

    return gregorian, jalali


def get_today():
    """
    Return today's Gregorian and Jalali dates.
    """

    gregorian = date.today()

    jalali = jdatetime.date.fromgregorian(
        date=gregorian
    )

    return gregorian, jalali


def jalali_doy(jdate):
    month_days = [
        31, 31, 31, 31, 31, 31,
        30, 30, 30, 30, 30, 29
    ]

    return (
        sum(month_days[:jdate.month - 1])
        + jdate.day
    )


def print_report(gregorian, jalali):

    g = get_details(gregorian)
    j = get_details(jalali)

    txt = [
        "Gregorian:",
        "-" * 40,
        f"Date       : {g.date}",
        f"Weekday    : {g.weekday}",
        f"Week       : {g.week}",
        f"Julian Day : {g.doy}",
        "",
        "Jalali:",
        "-" * 40,
        f"Date       : {j.date}",
        f"Weekday    : {j.weekday}",
        f"Week       : {j.week}",
        # f"Julian Day : {j.doy}",
        f"DOY        : {jalali_doy(jalali)}",
    ]
    print("\n".join(txt))


def main():

    parser = ArgumentParser(
        description="Gregorian/Jalali calendar utility"
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--jal",
        metavar="YYYY-MM-DD",
        help="Jalali date"
    )

    group.add_argument(
        "--greg",
        metavar="YYYY-MM-DD",
        help="Gregorian date"
    )

    args = parser.parse_args()

    if args.jal:
        gregorian, jalali = jalali_to_gregorian(
            args.jal
        )

    elif args.greg:
        gregorian, jalali = gregorian_to_jalali(
            args.greg
        )

    else:
        gregorian, jalali = get_today()

    print_report(
        gregorian,
        jalali
    )


if __name__ == "__main__":
    main()
